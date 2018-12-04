import numpy as np
import ipdb


def nms(scores, overlaps, th):
    idx = []
    N = len(scores)
    if len(scores) < 1:
        return idx
    assert N == overlaps.shape[0] and N == overlaps.shape[1]

    scores = np.array(scores)
    sidx = np.argsort(scores)[::-1]
    idx.append(sidx[0])  # retain the idx c.t. the highest score
    i = 0
    while i < N - 1:
        i += 1
        cIou = overlaps[idx, sidx[i]]
        if cIou.max() < th:
            # current obj overlaps less than the th with all selected objs
            idx.append(sidx[i])

    return idx


def apply_lin_regressor(boxes, linreg, maxWH):
    # augment boxes for bias and apply regression
    reg_boxes = np.concatenate((boxes, np.ones((boxes.shape[0], 1))), axis=1)
    reg_boxes = np.matmul(linreg, reg_boxes.transpose()).transpose()

    # clip boxes then ensure x1 <= x2 and y1 <= y2
    reg_boxes = clipboxes(reg_boxes, maxWH)
    reg_boxes[:, 2] = reg_boxes[:, [0, 2]].max(1)  # x2
    reg_boxes[:, 3] = reg_boxes[:, [1, 3]].max(1)  # y2

    check_boxes(reg_boxes)
    return reg_boxes


def st_overlap_tracksets(track_set1,
                         track_set2=None,
                         track_set2_iskeyframes=False,
                         recall=False):
    # if recall
    # trim the tracks to the gt time bounds before eval
    # get per_gtframe_sp: spatial overlap at each gt annotation

    if track_set2 == None:
        track_set2 = track_set1
        onself = True
    else:
        onself = False

    N1, N2 = len(track_set1), len(track_set2)
    t_ov = np.zeros((N1, N2), dtype=float)
    s_ov = np.zeros((N1, N2), dtype=float)
    st_ov = np.zeros((N1, N2), dtype=float)
    per_gtframe_sp = [[] for i in range(N1)]
    for i in range(N1):
        for j in range(N2):
            if not onself or j >= i:
                if track_set2_iskeyframes:
                    res = st_overlap_keyframes(
                        track_set1[i],
                        track_set2[j]['keyframes'],
                        trim2t2=recall,
                        perframe_sp=recall)
                else:
                    res = st_overlap(
                        track_set1[i],
                        track_set2[j],
                        trim2t2=recall,
                        perframe_sp=recall)

                if recall:
                    st_, s_, t_, _gtframe_sp = res
                else:
                    st_, s_, t_ = res

                st_ov[i, j], s_ov[i, j], t_ov[i, j] = st_, s_, t_
                st_ov[i, j], s_ov[i, j], t_ov[i, j] = st_, s_, t_
                if recall:
                    per_gtframe_sp[i].append(_gtframe_sp)
            else:
                st_ov[i, j], s_ov[i, j], t_ov[i, j] = st_ov[j, i], s_ov[
                    j, i], t_ov[j, i]
        if recall:
            if N2 > 0:
                per_gtframe_sp[i] = np.concatenate(per_gtframe_sp[i], 1)
            else:
                per_gtframe_sp[i] = np.zeros((1, 0))
    if recall:
        if N1 == 0:
            per_gtframe_sp = np.zeros((0, N2))
        else:
            per_gtframe_sp = np.concatenate(per_gtframe_sp, 0)

    if recall:
        return st_ov, s_ov, t_ov, per_gtframe_sp
    else:
        return st_ov, s_ov, t_ov


def st_recall_tracksets(track_set, track_keyframes):
    return st_overlap_tracksets(track_set, track_keyframes, recall=True)


def st_overlap_trackset_keyframes(track_set, track_keyframes):
    return st_overlap_tracksets(track_set, track_keyframes, True)


def st_recall_trackset_keyframes(track_set, track_keyframes):
    return st_overlap_tracksets(track_set, track_keyframes, True, recall=True)


def st_overlap(track1, track2, trim2t2=False, perframe_sp=False):
    t_ov, ov_frames = temporal_overlap(
        track1['tbound'], track2['tbound'], trim2t2=trim2t2)

    if perframe_sp:
        per_gtframe_sp = np.zeros((1, track2['N_frames']))

    s_ov = 0
    if ov_frames:
        # select overlapping boxes
        fbounds = (min(ov_frames), max(ov_frames))
        _, _, t1boxes = framebounds2idx(track1, fbounds)
        _, _, t2boxes = framebounds2idx(track2, fbounds)

        _ov = spatial_overlap(t1boxes, t2boxes)
        s_ov = _ov.mean()

        if perframe_sp:
            per_gtframe_sp[0, :len(
                _ov
            )] = _ov  # gt not temporally spanned stay at 0 (not ordered)

    if perframe_sp:
        return t_ov * s_ov, s_ov, t_ov, per_gtframe_sp
    return t_ov * s_ov, s_ov, t_ov


def st_overlap_keyframes(track, keyframes, trim2t2=False, perframe_sp=False):
    # if per_gtframe_sp: also return spatial overlap at each gt annotation
    t_ov, ov_frames = temporal_overlap(
        track['tbound'], keyframes['tbound'], trim2t2=trim2t2)
    for kframe in keyframes['keylist']:
        assert kframe['frame_num'] >= keyframes['tbound'][0]
        assert kframe['frame_num'] <= keyframes['tbound'][1]

    if perframe_sp:
        per_gtframe_sp = np.zeros((1, len(keyframes['keylist'])))

    s_ov = 0
    if ov_frames:
        numS = 0
        for k, kframe in enumerate(keyframes['keylist']):
            # select boxes at keyframe positions
            fnum = kframe['frame_num']
            if fnum in ov_frames:
                keybox = np.expand_dims(kframe['boxes'], 0)
                _, _, tbox = framebounds2idx(track, (fnum, fnum))
                _ov = spatial_overlap(keybox, tbox)[0]
                s_ov += _ov
                numS += 1
                if perframe_sp:
                    per_gtframe_sp[0, k] = _ov

        if numS > 0:
            s_ov /= numS

    if perframe_sp:
        return t_ov * s_ov, s_ov, t_ov, per_gtframe_sp
    return t_ov * s_ov, s_ov, t_ov


def framebounds2idx(track, fbound):
    # from frame bounds starting at 1, returns the corresponding indices and boxes
    checkFramebounds(track, fbound)

    fstart, fend = fbound
    t_start = track['tbound'][0]  # first frame of the track
    t_1, t_2 = fstart - t_start, fend - t_start  # get the correspondance in the track position (0 based)

    boxes = track['boxes'][t_1:t_2 + 1, :]

    checkIdx(track, t_1, t_2)
    return t_1, t_2, boxes


def idx2framebounds(track, t_1, t_2):
    # from zero based start and end indices in the track, return the frame bounds and boxes
    checkIdx(track, t_1, t_2)

    t_start = track['tbound'][0]  # first frame of the track
    fstart, fend = t_start + t_1, t_start + t_2

    fbound = (fstart, fend)
    boxes = track['boxes'][t_1:t_2 + 1, :]

    checkFramebounds(track, fbound)
    return fbound, boxes


def checkIdx(track, t_1, t_2):
    assert t_1 <= t_2
    assert t_1 >= 0
    assert t_2 < track['N_frames']
    assert track['boxes'].shape[0] == track['N_frames']


def checkFramebounds(track, fbound):
    fstart, fend = fbound
    assert fstart <= fend
    assert fstart >= track['tbound'][0]
    assert fend <= track['tbound'][1]
    assert track['tbound'][0] >= 1
    assert track['boxes'].shape[0] == track['N_frames']
    assert track['tbound'][1] == track['N_frames'] + track['tbound'][0] - 1


def temporal_overlap(t1_bounds, t2_bounds, trim2t2=False):
    # given start/end position of tracks t1 and t1
    # return the temporal overlap and the overlaping frame numbers
    # if trim2t2, reduce t1_bounds to its intersection with t2_bounds (useful for recall computation)
    t1_start, t1_end = t1_bounds
    t2_start, t2_end = t2_bounds
    assert t1_start > 0 and t2_start > 0
    assert t1_start <= t1_end
    assert t2_start <= t2_end

    max_start = max(t1_start, t2_start)
    min_end = min(t1_end, t2_end)

    if max_start > min_end:
        return 0, []

    if trim2t2:
        t1_start = max_start
        t1_end = min_end

    t1_len = t1_end - t1_start + 1
    t2_len = t2_end - t2_start + 1
    inter = min_end - max_start + 1
    union = t1_len + t2_len - inter

    t_ov = float(inter) / union
    return t_ov, range(max_start, min_end + 1)


def check_boxes(box_set1, box_set2=None):
    # check that boxes have 4 coordinates
    # check no value is <1 (coordinates are supposed to start at 1)
    # If there are 2 sets, check they have the same number of boxes
    # Be carreful, it does not necessary ensure that boxes are in x1,y1,x2,y2 format
    # e.g. if w >= x1 the box will pass the check
    # return the number of boxes

    N = box_set1.shape[0]
    assert N > 0
    assert box_set1.shape[1] == 4, 'Boxes do not have 4 coordinates'
    assert not (box_set1 < 1).sum(), 'Some coordinates are below 1'
    assert not (box_set1[:, 0] >
                box_set1[:, 2]).sum(), 'Boxes are not in x1,y1,x2,y2 format'
    assert not (box_set1[:, 1] >
                box_set1[:, 3]).sum(), 'Boxes are not in x1,y1,x2,y2 format'
    assert np.issubdtype(np.float, box_set1.dtype) or np.issubdtype(
        np.float32, box_set1.dtype), (
            'Boxes are not in floating format: conversion can go wrong!')
    if not box_set2 is None:
        assert N == box_set2.shape[
            0], 'the 2 sets do not contain the same number of bboxes'
        check_boxes(box_set2)

    return N


def spatial_intersetion(box_set1, box_set2):
    # return the element-wise box spatial intersetion between the two set of size (N,4)
    # Boxes are in x1,y1,x2,y2 format
    check_boxes(box_set1, box_set2)
    leftBound = np.maximum(box_set1[:, 0], box_set2[:, 0])
    topBound = np.maximum(box_set1[:, 1], box_set2[:, 1])
    rightBound = np.minimum(box_set1[:, 2], box_set2[:, 2])
    botomBound = np.minimum(box_set1[:, 3], box_set2[:, 3])

    int_area = np.multiply(rightBound - leftBound + 1,
                           botomBound - topBound + 1)
    no_ov = np.logical_or(leftBound > rightBound, topBound > botomBound)
    int_area[no_ov] = 0

    return int_area


def clipboxes(boxes, maxWH):
    maxW, maxH = maxWH
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(1, maxW)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(1, maxH)
    check_boxes(boxes)
    return boxes


def scaleboxes_(boxes, sc):
    # inline
    sc = float(sc)
    W, H = get_WH(boxes)  # this also check boxes
    d_W = W * (sc - 1) / 2
    d_H = H * (sc - 1) / 2

    boxes[:, 0] -= d_W
    boxes[:, 1] -= d_H
    boxes[:, 2] += d_W
    boxes[:, 3] += d_H

    if sc < 1:
        # boxes should not have x1 (y1) > x2 (y2): set to the middle point
        _av = (boxes[:, 0] + boxes[:, 2]) / 2
        idx = boxes[:, 0] > boxes[:, 2]
        boxes[idx, 0] = _av[idx]
        boxes[idx, 2] = _av[idx]
        _av = (boxes[:, 1] + boxes[:, 3]) / 2
        idx = boxes[:, 1] > boxes[:, 3]
        boxes[idx, 1] = _av[idx]
        boxes[idx, 3] = _av[idx]


def get_WH(box_set1):
    # return the width W (resp. height H) of the box set from the x1,x2 (resp. y1,y2) coordinates
    check_boxes(box_set1)
    W_set1 = box_set1[:, 2] - box_set1[:, 0] + 1
    H_set1 = box_set1[:, 3] - box_set1[:, 1] + 1
    return W_set1, H_set1


def get_area(box_set1):
    # return the set box areas
    W_set1, H_set1 = get_WH(box_set1)
    area = np.multiply(W_set1, H_set1)
    assert not (area <= 0).sum(), 'Box area is <=0'
    return area


def spatial_overlap(box_set1, box_set2):
    # return the element-wise box spatial overlap (IoU) between the two set of size (N,4)
    # Boxes are in x1,y1,x2,y2 format
    check_boxes(box_set1, box_set2)
    int_area = spatial_intersetion(box_set1, box_set2)
    intersection_over_detection = False
    if intersection_over_detection:
        uni_area = get_area(box_set1)
    else:
        uni_area = get_area(box_set1) + get_area(box_set2) - int_area
    return np.divide(int_area, uni_area)


def get_evaluation_args():
    parser = ArgumentParser(description='Train action localization model')
    parser.add_argument('-expname', type=str)


def scale_pad_boxes(boxes, sc_padW_padH):
    # scale THEN pad the set of boxes
    check_boxes(boxes)
    boxes = np.array(boxes)
    sc, padW, padH = sc_padW_padH
    assert sc > 0 and padW >= 0 and padH >= 0
    boxes = boxes * sc
    boxes[:, [0, 2]] += float(padW) / 2
    boxes[:, [1, 3]] += float(padH) / 2

    return boxes


def spot_inside_box(spot, box):
    return spot[0] > box[0] and spot[0] < box[2] and spot[1] > box[1] and spot[
        1] < box[3]


def get_center_box(box):
    return [0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])]


def dist_spots(spot1, spot2):
    return np.linalg.norm(np.array(spot1) - np.array(spot2))
