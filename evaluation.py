import track_utils as utils
import pickle
import re
import os
import scipy.signal as signal
import multiprocessing
import time
import numpy as np
import copy
import ipdb
import collections
import glob

global global_tracks


def smooth_padding(scores, smooth_window, smoothing):
    # define the window size in which we pool both start/end pad values
    pad_window = max(min(smooth_window, 10), 1)

    # get start and end pad values
    sz = min(pad_window, scores.shape[0])
    pad_start = np.median(scores[:sz])
    pad_end = np.median(scores[-sz:])

    # pad original scores
    scores_padded = np.concatenate((np.repeat(pad_start, smooth_window),
                                    scores, np.repeat(pad_end, smooth_window)))

    # apply smoothing and truncate to the original size
    if smoothing == 'median':
        smoothed = signal.medfilt(scores_padded, smooth_window)
    elif smoothing == 'average':
        smoothed = np.convolve(
            scores_padded,
            np.ones((smooth_window, )) / smooth_window,
            mode='same')
    return smoothed[smooth_window:-smooth_window]


def _process_scores(args):
    tracks, smooth_window, smoothing, tscore_th = args
    del_idx = []
    for idx, track in enumerate(tracks):

        if not tscore_th is None:
            if track['det_track_score'] < tscore_th:
                del_idx.append(idx)  # rm track with score below th
                continue
        if smooth_window > 1:
            track['scores'] = smooth_padding(track['scores'], smooth_window,
                                             smoothing)

    for idx in del_idx[::-1]:
        del tracks[idx]

    return tracks


def track_scoring(scores, N=40):
    # compute on top N scores
    scores = np.sort(scores)[::-1]  # sort by descending order
    return scores[:N].mean()


def tpfp(iou_matrix, th, ignoreGT=None):
    nDet, nGT = iou_matrix.shape
    tp = np.zeros(nDet, dtype=bool)
    fp = np.zeros(nDet, dtype=bool)
    if ignoreGT is None:
        ignoreGT = np.zeros(nGT, dtype=bool)
    else:
        assert len(ignoreGT) == nGT

    best_gt = iou_matrix.argmax(1)
    best_iou = iou_matrix[range(len(best_gt)), best_gt]
    gt_covered = np.zeros(nGT, dtype=bool)

    for i in range(nDet):
        if best_iou[i] >= th:
            i_best = best_gt[i]
            if not ignoreGT[i_best]:
                if gt_covered[i_best]:
                    fp[i] = 1  # duplicate
                else:
                    tp[i] = 1
                    gt_covered[i_best] = 1
        else:
            fp[i] = 1

    return tp, fp


def ap(rec, prec):
    # From Online Real-time Multiple Spatiotemporal Action Localisation and Prediction, G. Singh et al. (ICCV 2007)
    # and Deep Learning for Detecting Multiple Space-Time Action Tubes in Videos, S.Saha et al. (BMVC16)
    # following the PASCAL VOC 2011 devkit
    if rec.shape[0] == 0 or prec.shape[0] == 0:
        return 0

    if prec[0] == 0:
        assert rec[0] == 0
    else:
        assert prec[0] == 1
    if rec[0] > 0: assert prec[0] == 1
    else: assert prec[0] == 0

    # compute the precision envelope
    # interpolate precision: given a recall r
    # retain the highest precision at recall >= r
    prec = np.maximum.accumulate(prec[::-1])[::-1]

    # insert dummy 0 for first delta
    rec = np.insert(rec, 0, 0)
    prec = np.insert(prec, 0, 0)

    # to calculate area under PR curve, look for cutoff points
    # where X axis (recall) changes value
    cut = rec[1:] != rec[:-1]

    # and sum (\Delta recall) * prec
    ap = ((rec[1:][cut] - rec[:-1][cut]) * prec[1:][cut]).sum()

    return ap


def _temporal_localization(args):
    tracks, loc_th, min_length = args

    new_tracks = []
    optional_fields = [
        'subname'
    ]  # copy these fields to the subtracks if they exist in the track
    for track in tracks:
        # init
        scores = track['scores']
        T = track['N_frames']
        start_frame = track['tbound'][0]  # frame count starts at 1
        v = track['videoname']
        assert scores.shape[0] == T
        track_used = np.zeros(track['N_frames'], dtype=bool)

        positions = np.array(range(track['N_frames']), dtype=long)
        while ((~track_used).any()):
            max_pos = scores[~track_used].argmax(
            )  # get max score on remaining frames
            max_pos = positions[~track_used][
                max_pos]  # get the position on the whole track
            max_value = scores[max_pos]

            if max_value >= loc_th:
                _start, _end = max_pos, max_pos

                while _end < T - 1 and scores[_end + 1] >= loc_th:
                    _end += 1
                while _start > 0 and scores[_start - 1] >= loc_th:
                    _start -= 1
            else:
                break
            track_used[_start:_end + 1] = True
            tlen = _end - _start + 1

            if tlen >= min_length:
                new_t = {
                    'videoname': v,
                    'N_frames': tlen,
                    'tbound': (_start + start_frame, _end + start_frame)
                }
                for ff in ['boxes', 'scores']:
                    new_t[ff] = track[ff][_start:_end + 1]
                for ff in optional_fields:
                    if ff in track:
                        new_t[ff] = track[ff]
                new_t['track_score'] = track_scoring(new_t['scores'])

                new_tracks.append(new_t)
    return new_tracks


def _nms(args):
    tracks, nms = args
    iou_matrix, _, _ = utils.st_overlap_tracksets(tracks)
    t_scores = [x['track_score'] for x in tracks]
    idx = utils.nms(t_scores, iou_matrix, nms)

    return [tracks[t] for t in idx]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    nclasses = x.shape[1]
    _max = np.repeat(np.max(x, 1)[:, None], nclasses,
                     1)  # for numrical stability
    e_x = np.exp(x - _max)
    _sum = np.repeat(e_x.sum(1)[:, None], nclasses, 1)
    return np.divide(e_x, _sum)


def _loadvideotracks(args):
    v, trackpath, nclasses, regress, linear_regressor, track_class_agnostic, scale, normalization_fn = args
    if not linear_regressor is None:
        assert not regress
    proc = multiprocessing.current_process().name
    if proc == "PoolWorker-1":
        print v, proc

    vidpath = '%s/%s' % (trackpath[0], v)
    numExp = len(trackpath)
    orignal_file_path = vidpath + '.pkl'
    if os.path.exists(orignal_file_path):
        # the original track path has been detected (tracks are not scored) useful for recall computation
        from_original_file = True
        with open(orignal_file_path) as f:
            original_file = pickle.load(f)
        ntracks = len(original_file['tracks'])
        assert numExp == 1
    else:
        from_original_file = False
        if os.path.exists(vidpath):
            tracklist = os.listdir(vidpath)
            ntracks = len(tracklist)
        else:
            ntracks = 0
    if ntracks == 0:
        print 'Warning: no intput track found in %s' % vidpath

    loadedtracks = [{v: []} for i in range(nclasses)]
    for t in range(ntracks):
        for exp in range(numExp):
            if from_original_file:
                tfile = original_file['tracks'][t]
                tfile['scores'] = np.zeros([tfile['N_frames'],
                                            nclasses + 1])  # make dummy scores
            else:
                vidpath = '%s/%s' % (trackpath[exp], v)
                tpath = '%s/track%05d.pkl' % (vidpath, t + 1)

                with open(tpath) as f:
                    tfile = pickle.load(f)

            nframes = tfile['tbound'][1] - tfile['tbound'][0] + 1
            assert nframes == tfile['boxes'].shape[0] and nframes == tfile[
                'N_frames']

            if 'box_label' in tfile:
                tlabel = tfile['box_label']
            else:
                assert track_class_agnostic, 'if there is no label for the track we must be in class agnostic mode'
                tlabel = -1

            if not track_class_agnostic:
                assert tlabel > 0, 'if we do not consider all tracks (not class agnostic) then tracks have to be labeled'

            allscore = tfile['scores']
            if not (type(tfile['scores']) == np.ndarray):
                allscore = allscore.numpy()

            if normalization_fn is not None:
                if normalization_fn == 'softmax':
                    allscore = softmax(allscore)
                else:
                    raise ValueError('unknown normalization function')

            for c in range(nclasses):
                if (not track_class_agnostic and tlabel != c + 1):
                    continue
                cscore = allscore[:, c] / numExp

                if cscore.shape[0] != nframes:
                    # there is one score per feature, we have to duplicate
                    f2c = tfile['frame2chunk']
                    nChunks = cscore.shape[0]
                    assert nChunks == (f2c[-1] - f2c[0] +
                                       1), 'there is not one pred per chunk!'
                    c0 = f2c[0]
                    _dupl = [(i_c + c0 == f2c).sum() for i_c in range(nChunks)]
                    cscore = np.repeat(cscore, _dupl)
                    assert cscore.shape[0] == nframes

                if exp == 0:
                    track = {}
                    fields = ['N_frames', 'tbound']
                    if regress:
                        track['boxes'] = tfile['reg_boxes'][c]
                    elif not linear_regressor is None:
                        track['boxes'] = utils.apply_lin_regressor(
                            tfile['boxes'], linear_regressor[c],
                            tfile['WH_size'])
                    else:
                        fields.append('boxes')
                    if 'track_score' in tfile:
                        track['det_track_score'] = tfile['track_score']

                    for ff in fields:
                        track[ff] = tfile[ff]
                    track['videoname'] = v
                    track['scores'] = cscore
                    loadedtracks[c][v].append(track)
                    cmpboxes = tfile['boxes']

                    if scale > 0:
                        # scale boxes if one scale has been passed
                        boxes = np.array(track['boxes'])  # clone it
                        utils.scaleboxes_(boxes, scale)
                        if 'WH_size' in tfile:
                            WHmax = tfile['WH_size']
                        else:
                            WHmax = (320, 240)
                        boxes = utils.clipboxes(boxes, WHmax)
                        track['boxes'] = boxes
                else:
                    track = loadedtracks[c][v][-1]
                    track['scores'] += cscore
                    assert (cmpboxes - tfile['boxes']).sum(
                    ) == 0, 'Tracks to combine do not have same boxes'

    return loadedtracks


def append_tpfp(tp,
                fp,
                scores,
                tracks,
                gttracks,
                i_det,
                iou_th,
                on_keyframes,
                overlap_only=False,
                eval_recall=False):
    if eval_recall:
        assert overlap_only

    N = len(tracks)

    if overlap_only:
        t_scores = [1 for x in tracks]
    else:
        t_scores = [x['track_score'] for x in tracks]

    if gttracks:
        if on_keyframes:
            ignoreGT = np.zeros(len(gttracks), dtype=bool)
            for ig, cgt in enumerate(gttracks):
                if cgt['isAmbiguous']:
                    ignoreGT[ig] = True

            if eval_recall:
                sub_gttracks = []  # skip ignored
                for i_gt, skip in enumerate(ignoreGT):
                    if not skip:
                        sub_gttracks.append(gttracks[i_gt])
                iou_matrix, _, _, per_gtframe_sp = utils.st_recall_trackset_keyframes(
                    tracks, sub_gttracks)

            else:
                iou_matrix, _, _ = utils.st_overlap_trackset_keyframes(
                    tracks, gttracks)
        else:
            ignoreGT = None
            if eval_recall:
                iou_matrix, _, _, per_gtframe_sp = utils.st_recall_tracksets(
                    tracks, gttracks)

            else:
                iou_matrix, _, _ = utils.st_overlap_tracksets(tracks, gttracks)

        if eval_recall:
            n_annot = per_gtframe_sp.shape[1]
            if N == 0:
                return 0, 0, n_annot
            max_iou_per_gt = iou_matrix.max(0)
            max_iou_per_annot = per_gtframe_sp.max(0)
            recalled_gt = (max_iou_per_gt >= iou_th).sum()
            recalled_annot = (max_iou_per_annot >= iou_th).sum()
            return recalled_gt, recalled_annot, n_annot

        if overlap_only:
            return 0, iou_matrix

        t_tp, t_fp = tpfp(iou_matrix, iou_th, ignoreGT)

    else:  # all false if no GT of this class
        if eval_recall:
            return 0, 0, 0
        assert not overlap_only
        t_tp, t_fp = False, True

    tp[i_det:i_det + N] = t_tp
    fp[i_det:i_det + N] = t_fp
    scores[i_det:i_det + N] = t_scores

    return i_det + N


def get_class_gt(gts, c):
    # get GT for the current class
    gttracks = []
    for g in gts['gts']:
        if g['label'] == c + 1:
            gttracks.append(g)
    return gttracks


def get_ap(tp, fp, scores, i_det, num_positives):
    # truncate
    tp = tp[:i_det]
    fp = fp[:i_det]
    scores = scores[:i_det]

    # sort according to score
    sidx = scores.argsort()[::-1]
    tp = tp[sidx]
    fp = fp[sidx]
    scores = scores[sidx]

    # AP
    tp = tp.cumsum(dtype=float)
    fp = fp.cumsum()
    if len(fp) > 0 and fp[0] == 0 and tp[0] == 0:
        # at least the first detection was assigned to a ignored GT
        _ze = np.logical_and(fp == 0, tp == 0)
        _max_ze = len(_ze) - _ze[::-1].argmax() - 1
        assert _ze[:_max_ze + 1].sum() == _max_ze + 1
        # ignore these firsts elements
        tp = tp[_max_ze + 1:]
        fp = fp[_max_ze + 1:]
        scores = scores[_max_ze + 1:]

    rec = tp / num_positives
    if num_positives == 0:
        print 'WARNING THERE ARE NO POSITIVES IN GT!'
        rec[:] = 0
    prec = tp / (fp + tp)
    c_ap = ap(rec, prec)

    # recall
    recv = 0 if rec.shape[0] == 0 else rec[-1]

    issame = scores.shape[0] > 1 and scores[0] == scores[
        1]  # check if (first) scores are exactly equal

    return c_ap, recv, issame


def get_num_positives(gttracks):
    ngts = len(gttracks)
    for cgt in gttracks:
        if 'isAmbiguous' in cgt and cgt['isAmbiguous']:
            ngts -= 1
    return ngts


def _sub_st_loc_map(args):
    testlist, iou_th, i_th, nclasses, gt, on_keyframes, eval_recall = args

    testtracks = global_tracks[i_th]
    _MAXDET = int(1e3 * len(testlist))

    log_str = ''
    proc = multiprocessing.current_process().name
    log_str += '\n\nST-loc @IoU=%.02f' % (iou_th)
    if eval_recall:
        final_str = '\nMEAN:\n ST-RECALL = %.3f / ANNOT-RECALL = %.3f'
    else:
        final_str = '\nmAP = %.3f / rec = %.3f'
    aps = np.zeros(nclasses)
    recs = np.zeros(nclasses)
    for c in range(nclasses):
        nodetstr = ''
        num_positives = 0
        num_annotations = 0
        ctracks = testtracks[c]
        tp = np.zeros(_MAXDET, dtype=bool)
        fp = np.zeros(_MAXDET, dtype=bool)
        scores = np.zeros(_MAXDET, dtype=float)
        i_det = 0  # next idx to fill
        for v in testlist:
            gts = gt[v]

            assert len(gts['gts']
                       ) > 0, 'This was supposed to be handled in loadtracks'

            gttracks = get_class_gt(gts, c)
            ngts = get_num_positives(gttracks)
            num_positives += ngts

            if not v in ctracks:
                if ngts > 0:
                    nodetstr += '%s (%d GTs) ' % (v, ngts)
                continue

            tracks = testtracks[c][v]

            ntracks = len(tracks)
            prev_i_det = i_det

            if eval_recall:
                recalled_gt, recalled_annot, n_annot = append_tpfp(
                    tp,
                    fp,
                    scores,
                    tracks,
                    gttracks,
                    i_det,
                    iou_th,
                    on_keyframes,
                    overlap_only=True,
                    eval_recall=eval_recall)
                num_annotations += n_annot
                aps[c] += recalled_gt
                recs[c] += recalled_annot
                continue

            i_det = append_tpfp(tp, fp, scores, tracks, gttracks, i_det,
                                iou_th, on_keyframes)

        assert num_positives > 0, 'No GT for class %d' % (c + 1)

        if eval_recall:
            aps[c] = aps[c] / num_positives  # ST recall
            recs[c] = recs[c] / num_annotations  # annot recall

            log_str += '\nClass %d ST-recall = %.3f - Annot-recall = %.3f' % (
                c + 1, aps[c], recs[c])

            continue

        c_ap, rec, issame = get_ap(tp, fp, scores, i_det, num_positives)
        aps[c] = c_ap
        recs[c] = rec

        # disp info
        if nodetstr != '':
            log_str += '\nNo detection for %s' % (nodetstr)
        if issame:
            log_str += '\nWARNING SOME SCORES ARE EXACTLY EQUAL, ORDER MIGHT MATTER!'

        # compute nb of FP in the first 50% of TP
        s_tp = tp[:i_det]
        tp_idx = np.linspace(0, i_det, i_det, dtype=int)
        tp_idx = tp_idx[s_tp]
        tpnum = len(tp_idx)
        if tpnum >= 50:
            fp_at_50 = fp[:tp_idx[49]].sum()  # FP in first TOP50
        else:
            fp_at_50 = fp.sum()
        log_str += '\nClass %d AP = %.3f (max recall = %.3f) - P/TP/FP/FP@50 %d/%d/%d/%d' % (
            c + 1, c_ap, rec, num_positives, tpnum, fp.sum(), fp_at_50)
    log_str += final_str % (aps.mean(), recs.mean())
    return iou_th, i_th, aps, log_str


class Evaluation():
    def __init__(self,
                 datasetname,
                 exppath,
                 testlistpath,
                 iou,
                 loc_th=0.1,
                 smooth_window=-1,
                 nms=0.2,
                 nthreads=5,
                 min_length=22,
                 regress=False,
                 smoothing='median',
                 track_class_agnostic=False,
                 force_no_regressor=False,
                 tscore_th=None,
                 eval_recall=False,
                 from_original_track_files=None,
                 scale=-1,
                 normalization_fn=None,
                 one_th_per_iou=False,
                 cachedir='.'):
        # params specific to datasets
        if datasetname == 'UCF101':
            self.nclasses = 24
            self.gtpath = '/sequoia/data2/gcheron/UCF101/detection/gtfile.py'
            self.on_keyframes = False
            self.linear_regressor = None
        elif datasetname == 'DALY':
            self.nclasses = 10
            self.gtpath = '/sequoia/data2/gcheron/DALY/gtfile.pkl'
            self.on_keyframes = True
            if force_no_regressor:
                self.linear_regressor = None
            else:
                with open('/sequoia/data2/gcheron/DALY/reg_matrices.pkl',
                          'r') as f:
                    self.linear_regressor = np.load(f)
        else:
            raise ValueError('Unknown dataset %s', (datasetname))

        self.tscore_th = tscore_th  # threshold for a track to be selected

        self.track_class_agnostic = track_class_agnostic  # for each class, consider tracks from all detection classes

        assert not (regress and (not self.linear_regressor is None))

        # if eval_recall:
        # if on_keyframes, get the % of keyframes recovered by the input tracks @ at spatial th: iou_th
        # otherwise, get the % of spatio-temporal GT intervals recovered by the input tracks @ at S-T th: iou_th
        self.eval_recall = eval_recall

        self.scale = scale  # scale boxes by this factor if > 0
        self.normalization_fn = normalization_fn

        # data paths
        self.datasetname = datasetname
        self.from_original_track_files = from_original_track_files
        if self.from_original_track_files is not None:
            exppath = self.from_original_track_files

        if type(exppath) != list:
            self.trackpath = [exppath]
        else:
            self.trackpath = exppath

        self.cachedir = cachedir
        self.cachepath = self.cachedir + '/evaluation_cache/'
        for i in range(len(self.trackpath)):
            expname = re.sub('(.*[^/])(.*)', r'\1',
                             self.trackpath[i])  # remove eventual last /
            expname = re.sub('.*/', '', expname)
            self.cachepath += '__++' + expname
            if from_original_track_files is None:
                self.trackpath[i] += '/tracks'
        self.testlistpath = testlistpath

        # track post-processing params
        self.loc_th = loc_th  # array of localization th (D x N), N number of sets
        if type(self.loc_th) == np.ndarray:
            assert self.loc_th.ndim == 2
            if self.loc_th.shape[0] == 1:
                self.loc_th = self.loc_th.repeat(self.nclasses, 0)
            else:
                assert self.loc_th.shape[0] == self.nclasses
        else:  # this is just a scalar
            self.loc_th = np.zeros((self.nclasses, 1))
            self.loc_th[:] = loc_th

        self.smooth_window = smooth_window
        self.smoothing = smoothing
        self.nms = nms
        self.min_length = min_length

        # eval params
        self.iou = iou  # list of different iou of evaluation
        if type(self.iou) != list:
            self.iou = [self.iou]
        self.ap = np.zeros(
            (len(self.iou), self.loc_th.shape[1], self.nclasses)) - 1

        self.one_th_per_iou = one_th_per_iou  # evaluate only on th list per iou (loc_th[i] ---> iou[i])
        if self.one_th_per_iou:
            assert self.loc_th.shape[1] == len(self.iou)

        # other
        self.nthreads = nthreads
        self.regress = regress  # if there are several track paths (exps) the regression from the first one is considered

        with open(self.gtpath) as f:
            self.gt = pickle.load(f)
        #exec("for v in self.gt:\n\tfor t in self.gt[v]['gts']:\n\t\tt['boxes']-=1") # ONE LINE DEBUG :)

        if self.from_original_track_files is not None:
            assert self.eval_recall

        if self.eval_recall:
            # do not need it
            self.loc_th = self.loc_th[:, 0, None]  # keep only one
            self.loc_th[:] = -1

        # check params
        if isinstance(loc_th, list):
            assert len(loc_lh) == self.nclasses

    def eval(self):
        c_time = time.time()

        self.loadtracks()
        print 'Loading time %d s' % (time.time() - c_time)
        c_time = time.time()

        if self.eval_recall:
            self.testtracks = []
            self.testtracks.append(
                self.loadedtracks)  # mimic testtracks at only on det th
        else:
            self.post_process_tracks()
        print 'Post proc time %d s' % (time.time() - c_time)

        c_time = time.time()
        calibration = self.st_loc_map()
        print 'mAP time %d s' % (time.time() - c_time)
        print self.trackpath

        return calibration

    def get_test_list(self):
        # get test list
        vlist = []
        self.testlist = []
        with open(self.testlistpath) as f:
            vlist = f.readlines()
        vlist = [re.sub(' .*', '', x.strip()) for x in vlist]

        for v in vlist:
            if self.gt[v]['N_gts'] < 1:
                print 'Discard %s (no GT available)' % v
                continue
            self.testlist.append(v)

    def loadtracks(self):

        self.load_cache = self.cachepath + '/tracks_%s_N-1_reg%d' % (re.sub(
            '.*/', '', self.testlistpath), self.regress)
        if not self.linear_regressor is None:
            self.load_cache += '_linreg'
        if self.track_class_agnostic:
            self.load_cache += '_trackCAgno'
        if self.scale > 0:
            self.load_cache += '_sc%.3f' % self.scale
        if self.normalization_fn is not None:
            self.load_cache += '_' + self.normalization_fn
        this_cache = self.load_cache + '/loaded.pkl'
        cache_fields = ['testlist', 'loadedtracks']

        if os.path.exists(this_cache):
            # load cache
            print 'loading file: %s' % this_cache
            with open(this_cache, 'rb') as f:
                cachefile = pickle.load(f)
            for ff in cache_fields:
                setattr(self, ff, cachefile[ff])
            t_total = cachefile['t_total']

        else:
            self.get_test_list()  # get test list

            # multithreading: load track files
            tlist = self.testlist
            #tlist=tlist[0:15] # DEBUG
            #for v in tlist:
            #   _loadvideotracks((v, self.trackpath, self.nclasses, self.regress, self.linear_regressor,
            #                              self.track_class_agnostic, self.scale, self.normalization_fn))
            res = self.run_multiprocessing(
                _loadvideotracks,
                [(v, self.trackpath, self.nclasses, self.regress,
                  self.linear_regressor, self.track_class_agnostic, self.scale,
                  self.normalization_fn) for v in tlist])
            # reorder tracks
            t_total = 0
            d_total = 0
            self.loadedtracks = [{} for i in range(self.nclasses)]
            for vid in res:
                for c in range(self.nclasses):
                    v = vid[c].keys()
                    assert len(v) == 1
                    v = v[0]
                    assert not v in self.loadedtracks[c]
                    self.loadedtracks[c][v] = vid[c][v]
                    t_total += len(vid[c][v])

            # save cache
            if not os.path.exists(self.load_cache):
                os.makedirs(self.load_cache)
            cachefile = {}
            for ff in cache_fields:
                cachefile[ff] = getattr(self, ff)
            cachefile['t_total'] = t_total
            with open(this_cache, 'w') as f:
                pickle.dump(cachefile, f)

        d_total = self.getNumDets(self.loadedtracks)
        print '%d tracks have been loaded (%d detections)' % (t_total, d_total)

    def getNumDets(self, loadedtracks):
        d = 0
        for ctracks in loadedtracks:
            for vtracks in ctracks:
                for track in ctracks[vtracks]:
                    d += track['boxes'].shape[0]
        return d

    def post_process_tracks(self):
        self.testtracks = []
        self.pproc_cache = []
        for i_th in range(self.loc_th.shape[1]):
            self.sub_post_process_tracks(i_th)

    def get_sw(self, loc_ths):
        mw = np.array(loc_ths)
        mw[:] = self.smooth_window
        mw[loc_ths <= 0] = -1  # no smoothing
        return mw

    def sub_post_process_tracks(self, i_th):
        assert len(self.testtracks) == i_th
        loc_ths = self.loc_th[:, i_th]
        assert len(loc_ths) == self.nclasses

        mw = self.get_sw(loc_ths)

        # per-class cache
        this_cache = []
        str_cache = self.load_cache + '/pproc_nms%.4f' % (self.nms)
        if self.smoothing == 'median':
            smstr = 'med'
        else:
            smstr = self.smoothing
        if not self.tscore_th is None:
            str_cache += '_tsth%.3f' % self.tscore_th

        for c, lt in enumerate(loc_ths):
            cstr = '_Class%d_lt%.4f_%s%d' % (c, lt, smstr, mw[c])
            if lt > 0:
                ml = self.min_length
            else:
                ml = -1
            cstr = str_cache + cstr + '_mlen%d' % ml
            this_cache.append(cstr)
        self.pproc_cache.append(this_cache)

        pproc_cache = []  # already post-proc tracks
        class_tracks = []  # tracks to post-proc
        self.testtracks.append([])  # final tracks
        some_missing = False
        for c in range(self.nclasses):
            self.testtracks[i_th].append({})
            pproc_cache.append([])
            c_cache = this_cache[c]
            if os.path.exists(c_cache):
                if i_th > 0 and c_cache == self.pproc_cache[0][c]:
                    # if the cache path is equal to the first one (not dependent of i_th)
                    pproc_cache[c] = self.testtracks[0][
                        c]  # take the same videos instead of loading again
                else:
                    with open(c_cache, 'rb') as f:
                        print 'loading %s' % (c_cache)
                        pproc_cache[c] = pickle.load(f)
                ctracks = []  # no need to post-proc
            else:
                ctracks = self.loadedtracks[c]
                some_missing = True
            class_tracks.append(ctracks)

        if some_missing:
            class_tracks = self.run_multiprocessing(
                _process_scores,
                [(ctracks[v], int(mw[c]), self.smoothing, self.tscore_th, c)
                 for c, ctracks in enumerate(class_tracks)
                 for v in ctracks], True, True)
            class_tracks = self.run_multiprocessing(
                _temporal_localization,
                [(v, loc_ths[c], self.min_length, c)
                 for c, ctracks in enumerate(class_tracks)
                 for v in ctracks], True, True)
            class_tracks = self.run_multiprocessing(
                _nms, [(v, self.nms, c)
                       for c, ctracks in enumerate(class_tracks)
                       for v in ctracks], True)  # no track1by1 for NMS

            for c, ctracks in enumerate(class_tracks):
                for vid in ctracks:
                    if len(vid) > 0:
                        v = vid[0]['videoname']
                        assert not v in self.testtracks[i_th][c]
                        self.testtracks[i_th][c][v] = vid

                # save cache
                c_cache = this_cache[c]
                if not os.path.exists(c_cache):
                    with open(c_cache, 'w') as f:
                        pickle.dump(self.testtracks[i_th][c], f)

        # merge eventual cache
        for c, ctracks in enumerate(pproc_cache):
            if len(ctracks) > 0:
                assert len(self.testtracks[i_th][c]) == 0
                self.testtracks[i_th][c] = ctracks

    def run_multiprocessing(self,
                            _fun,
                            arglist,
                            class_format=False,
                            track1by1=False):
        pool = multiprocessing.Pool(self.nthreads)
        if class_format:  # last argument is the class
            classes = [x[-1] for x in arglist]
            arglist = [x[:-1] for x in arglist]
        _la = len(arglist)

        if track1by1:
            # split all video tracks (create videos with only 1 track)
            _tmp = []
            vididx = [
            ]  # assign and idx to a video (same for each if its tracks)
            v_count = -1
            for V in arglist:
                v = V[0]
                assert type(
                    v
                ) == list, 'with track1by1, first arg is supposed to contain the tracks!'
                v_count += 1
                if len(v) == 0:  # add a dummy track for video with no tracks
                    _tmp.append(
                        tuple([[]] + list(V[1:])))  # copy video parameters
                    vididx.append(v_count)
                else:
                    for t in range(len(v)):  # for all video tracks
                        _tmp.append(
                            tuple([[v[t]]] +
                                  list(V[1:])))  # copy video parameters
                        vididx.append(v_count)
            arglist = _tmp

        res = pool.map(_fun, arglist)
        pool.terminate()
        pool.join()

        if track1by1:  # reshape
            _tmp = []
            assert type(
                res[0]
            ) == list, 'with track1by1, result is supposed to be a list of tracks'
            for i, tracks in enumerate(res):
                idx = vididx[i]
                if len(_tmp) <= idx:
                    _tmp.append([])
                for t in tracks:  # one track could generate several outputs
                    _tmp[idx].append(t)  # merge them to the video outputs
            res = _tmp

        _lr = len(res)
        assert _la == _lr, 'number of results (%d) is different from number of inputs (%d)' % (
            _lr, _la)
        if class_format:  # reshape
            _tmp = [[] for i in range(self.nclasses)]
            for i, c in enumerate(classes):
                _tmp[c].append(res[i])
            res = _tmp
            classes = np.array(classes)
            for i in range(self.nclasses):
                assert len(res[i]) == (classes == i).sum()

        return res

    def st_loc_map(self):
        global global_tracks
        global_tracks = self.testtracks

        n_iou = len(self.iou)
        n_th = self.loc_th.shape[1]

        all_aps = self.run_multiprocessing(
            _sub_st_loc_map,
            [(self.testlist, self.iou[i_iou], i_th, self.nclasses, self.gt,
              self.on_keyframes, self.eval_recall) for i_iou in range(n_iou)
             for i_th in range(n_th)])

        for i_iou in range(n_iou):
            for i_th in range(n_th):
                _iou, _ith, _aps, _log = all_aps[i_iou * n_th + i_th]
                assert (self.ap[i_iou, i_th, :] != -1).sum() == 0
                assert self.iou[i_iou] == _iou and i_th == _ith
                self.ap[i_iou, i_th, :] = _aps
                print _log

        if self.eval_recall:
            print '\nIoU / ST-RECALL'
            iou_str = ''
            strec_str = ''
            for i in range(n_iou):
                iou_str += '%.3f ' % self.iou[i]
                strec_str += '%.3f ' % self.ap[i, 0, :].mean()
            print iou_str
            print strec_str + '\n'
            return

        # print summary
        allth = not self.one_th_per_iou
        print '\n\n=============================='
        print 'Thresholds:'
        _str = ''
        for i_th in range(n_th):
            _str += '\nSet %d:' % (i_th)
            for c in range(self.nclasses):
                if c % 5 == 0:
                    _str += '\n'
                _str += '%d: %.1E - ' % (c, self.loc_th[c, i_th])
        print _str

        for i_iou in range(n_iou):
            th_means = self.ap[i_iou].mean(1)
            best_i_th = th_means.argmax()
            loc_th = self.loc_th[0, best_i_th]
            print '\nST-loc @IoU=%.02f --> Best mAP = %.3f [loc_th = %.3f (%d)] ' % (
                self.iou[i_iou], th_means[best_i_th], loc_th, best_i_th)

            # print ap per class
            for c in range(self.nclasses):
                _apstr = ''

                if allth:
                    # get ap from all TH
                    for i_th in range(n_th):
                        _apstr += '%.3f ' % (self.ap[i_iou, i_th, c])
                else:
                    # get ap from the corresponding TH
                    assert n_iou == n_th
                    _apstr += '%.3f ' % (self.ap[i_iou, i_iou, c])
                print _apstr

            # print mAP
            _mapstr = ''
            if allth:
                # get map from all TH
                for i_th in range(n_th):
                    _mapstr += '%.3f ' % (self.ap[i_iou, i_th, :].mean())
            else:
                # get map from the corresponding TH
                _mapstr += '%.3f ' % (self.ap[i_iou, i_iou, :].mean())

            print '|\n|___> mAP = %s' % (_mapstr)

        if n_th == 1:
            print '\nIoU / mAP'
            iou_str = ''
            map_str = ''
            for i in range(n_iou):
                iou_str += '%.3f ' % self.iou[i]
                map_str += '%.3f ' % self.ap[i, 0, :].mean()
            print iou_str
            print map_str + '\n'

        if allth:
            return self.run_validation()

    def track_postproc(self, multiclass_tracks, do_nms=True):
        assert self.loc_th.shape == (self.nclasses,
                                     1), 'only one loc th set must be defined'
        class_tracks = []  # tracks to post-proc

        mw = self.get_sw(self.loc_th)

        # multiclass_tracks: vid x tracks x (scores)classes (each tracks has info for all classes)
        class_vid_tracks = [{} for c in range(self.nclasses)]
        for vtracks in multiclass_tracks:
            for c, ctracks in vtracks.iteritems():
                class_vtracks = []
                if len(ctracks) > 0:
                    for track in ctracks:
                        vid = track['videoname']
                        if not vid in class_vid_tracks[c]:
                            class_vid_tracks[c][vid] = []
                        class_vtracks.append(track)
                    # add all tracks of vid for class c
                    class_tracks.append((class_vtracks, int(mw[c]),
                                         self.smoothing, self.tscore_th, c))

        class_tracks = self.run_multiprocessing(_process_scores, class_tracks,
                                                True, True)
        class_tracks = self.run_multiprocessing(
            _temporal_localization, [(v, self.loc_th[c], self.min_length, c)
                                     for c, ctracks in enumerate(class_tracks)
                                     for v in ctracks], True, True)
        if do_nms:
            class_tracks = self.run_multiprocessing(
                _nms, [(v, self.nms, c)
                       for c, ctracks in enumerate(class_tracks)
                       for v in ctracks], True)  # no track1by1 for NMS

        # fill 'class_vid_tracks' correspondances
        for c, ctracks in enumerate(class_tracks):
            for i, vid in enumerate(ctracks):
                if len(vid) > 0:
                    v = vid[0]['videoname']
                    if v in class_vid_tracks[c] and 'subname' in vid[0]:
                        # if subname field is there we append subtracks from all tracks of this video
                        class_vid_tracks[c][v] += vid
                    else:
                        # otherwise, only one set of subtracks is expected per video
                        assert not class_vid_tracks[c][v]
                        class_vid_tracks[c][
                            v] = vid  # note the videos with no track anymore are still evaluated

        return class_vid_tracks

    def add_tracks2eval(self,
                        class_vid_tracks,
                        i_dets,
                        class_npos,
                        tpfpscores,
                        on_keyframes,
                        overlap_only=False):
        iou = self.iou

        if overlap_only:
            ov = [{} for c in range(self.nclasses)]
            iou = [self.iou[0]]  # use only one dummy IoU

        for c in range(self.nclasses):
            ctracks = class_vid_tracks[c]

            for v in ctracks:
                i_det = i_dets[c]
                gts = self.gt[v]

                if len(gts['gts']) == 0:
                    if c == 0: print 'no GT: discard %s' % (v)
                    assert not overlap_only
                    continue

                gttracks = get_class_gt(gts, c)
                ngts = get_num_positives(gttracks)

                tracks = ctracks[v]

                if not overlap_only:
                    class_npos[c] += ngts
                elif ngts == 0 or not tracks:
                    ov[c][v] = np.zeros((len(tracks), ngts), dtype=float)
                    continue

                if not tracks:
                    #if ngts > 0:
                    #   print '%s (%d GTs) ' % (v,ngts)
                    continue

                for i in range(len(iou)):
                    if overlap_only:
                        tp, fp, scores = None, None, None
                    else:
                        tp, fp, scores = tpfpscores[i][c]
                    new_i_det, ov_ = append_tpfp(tp, fp, scores, tracks,
                                                 gttracks, i_det, iou[i],
                                                 on_keyframes, overlap_only)

                    if overlap_only:
                        ov[c][v] = ov_

                    if i == len(iou) - 1:
                        i_dets[c] = new_i_det
        if overlap_only:
            return ov

    def get_mAP(self, i_dets, class_npos, tpfpscores):
        aps = np.zeros((len(self.iou), self.nclasses))
        recs = np.zeros((len(self.iou), self.nclasses))
        for i in range(len(self.iou)):
            for c in range(self.nclasses):
                i_det = i_dets[c]
                num_positives = class_npos[c]
                tp, fp, scores = tpfpscores[i][c]
                aps[i][c], recs[i][c], _ = get_ap(tp, fp, scores, i_det,
                                                  num_positives)
        return aps, recs, self.iou

    def run_validation(self):
        n_iou = len(self.iou)
        calibration = {}
        print 'Per-class validation:'
        _map_str = ''
        for i_iou in range(n_iou):
            _str = ''
            aps, best_th = self.validate_perclass_th(i_iou)
            calibration[self.iou[i_iou]] = best_th
            _ap = aps.mean()
            _str += 'ST-loc @IoU=%.02f -- %.3f mAP\nTH:' % (self.iou[i_iou],
                                                            _ap)
            _map_str += '%.3f ' % _ap
            for th in best_th:
                _str += ' %.3f' % th
            print _str
        print _map_str
        return calibration

    def validate_perclass_th(self, i_iou):
        n_th = self.loc_th.shape[1]
        aps = self.ap[i_iou].max(0)
        best_th = self.ap[i_iou].argmax(0)
        best_th = self.loc_th[range(self.nclasses), best_th].tolist()

        return aps, best_th
