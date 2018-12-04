"""Main launcher for the WeakActionLocalizer code."""

import numpy as np
import bcfw_diffrac
import linear_solvers
import data_handler
import argparse
import eval_functions
import os
from tqdm import tqdm
from subprocess import call
import pickle
import evaluation
import re

BIAS_VALUE = 100


def load_feats_and_gts(
        mode,
        path_list,
        prepend_name,
        feat_type,
        group_by,
        sp_iou_thresh,
        cache_dir,
        path_tracks,
        path_info,
        n_actions):
    print('Dealing with features of {} set'.format(mode))
    if mode == 'val':
        list_file = '{}/OF_vidlist_{}1.txt'.format(path_list, 'valtrainRate10_train')
    else:
        list_file = '{}/OF_vidlist_{}1.txt'.format(path_list, mode)
    with open(list_file, 'r') as f_list:
        or_list_vid = f_list.readlines()
        or_list_vid = [x.strip().split(' ')[0] for x in or_list_vid]

    # For each video get the data.
    name_feat = 'features_{}_gb_{}_{}_{}_spth_{}.pkl'.format(prepend_name, group_by,
                                                             mode, feat_type, sp_iou_thresh)
    path_feat = os.path.join(cache_dir, name_feat)

    if os.path.isfile(path_feat):
        print 'load features from:\n {}'.format(path_feat)
        with open(path_feat) as f:
            feats, labels, list_vid = pickle.load(f)
    else:
        feats = []
        labels = []
        list_vid = []
        for id_vid, video_name in enumerate(tqdm(or_list_vid)):
            present, x, y = data_handler.get_data_for_video(
                video_name,
                feat_type=feat_type,
                sp_iou_thresh=sp_iou_thresh,
                groupby=group_by,
                path_tracks=path_tracks,
                path_info=path_info,
                n_actions=n_actions)
            if present:
                feats.append(x)
                labels.append(y)
                list_vid.append(video_name)

        print 'save features to:\n {}'.format(path_feat)
        with open(path_feat, 'wb') as f:
            pickle.dump((feats, labels, list_vid), f)

    # Check that loading is done correctly.
    for v in or_list_vid:
        if not v in list_vid:
            print 'Missing: {}'.format(v)

    for v in list_vid:
        assert v in or_list_vid, 'Wrong features were loaded: {} is not in the input list. Remove cache?'.format(v)

    dim = feats[0].shape[1]
    ntracklets = 0
    for ff in feats:
        assert ff.shape[1] == dim
        ntracklets += ff.shape[0]

    print('Got {} feature/label video sources:'.format(len(feats)))
    print('{} tracklets of dimension {}'.format(ntracklets, dim))

    return feats, labels, list_vid


def launcher(
        group_by,
        path_list='/sequoia/data2/gcheron/UCF101/detection',
        path_info='/sequoia/data2/gcheron/UCF101/detection/mytracksK5_600k',
        path_tracks='/sequoia/data2/gcheron/pytorch/diffrac_action_localization/UCF101/results/mytracksK5_600k/tracks/',
        prepend_name='UCF101_spatiotemporal_at_least_one_gb',
        feat_type='RGB+OPF',
        sp_iou_thresh=0.3,
        write_eval=False,
        n_iterations=10000,
        alpha=1e-4,
        n_actions=25,
        cache_dir='/sequoia/data2/jalayrac/nips2017weakpose/cache/',
        res_dir='/sequoia/data2/jalayrac/nips2017weakpose/',
        cstrs_name='at_least_one_per_instance_unit_time',
        exp_suffix='',
        rdm_seed=19,
        path_log_eval=None,
        save_pr_curves=False,
        eval_frequency=500,
        val_eval=False,
        video_eval_args=None,
        video_eval_only=False,
        calibrate=False,
        use_calibration=False,
        no_init=False,
        no_feat_init=False):

    np.random.seed(rdm_seed)

    bias_value = BIAS_VALUE
    assert bias_value == 100, 'This value is hard coded in evaluation'

    need_init = not (video_eval_only or no_init)
    need_feat_init = not (video_eval_only or no_feat_init)
    if calibrate:
        mode = 'val'
    else:
        mode = 'train'

    # Get the features and GT.
    if need_feat_init:
        feats_train, labels_train, list_vid = load_feats_and_gts(
            mode,
            path_list,
            prepend_name,
            feat_type,
            group_by,
            sp_iou_thresh,
            cache_dir,
            path_tracks,
            path_info,
            n_actions)

    validation_info = None
    if need_feat_init:
        if val_eval:
            feats_val, labels_val, _ = load_feats_and_gts(
                'test',
                path_list,
                prepend_name,
                feat_type,
                group_by,
                sp_iou_thresh,
                cache_dir,
                path_tracks,
                path_info,
                n_actions)
            validation_info = {'gts': labels_val, 'feats': feats_val}

    # Get the at least one constraints for the training set.
    print('Dealing with constraints for {} set'.format(mode))
    # For each video get the data.

    name_cstrs = 'cstrs_{}_{}_gb_{}_spth_{}_{}'.format(
        prepend_name, cstrs_name, group_by, sp_iou_thresh, mode)
    path_cstrs = os.path.join(cache_dir, name_cstrs + '.npy')

    fn_cstrs = 'get_{}_cstr_for_video'.format(cstrs_name)
    compute_cstrs = True
    if not need_init:
        compute_cstrs = False
    elif os.path.isfile(path_cstrs):
        cstrs_train = np.load(path_cstrs).tolist()
        compute_cstrs = False

    if compute_cstrs:
        cstrs = []
        for id_vid, video_name in enumerate(tqdm(list_vid)):
            present, constraint = data_handler.get_general_cstr_for_video(
                video_name,
                groupby=group_by,
                path_info=path_info,
                path_tracks=path_tracks,
                n_actions=n_actions,
                constr_fn_name=fn_cstrs)
            if present:
                cstrs.append(constraint)
            assert present, (
                '{} is missing: please provide a list were all videos have gt/features/labels...'
            ).format(video_name)

        print('Saving constraints...')
        np.save(path_cstrs, cstrs)
        cstrs_train = cstrs

    # Generate random init.
    print('Generating a random Y init...')
    asgn_train = []

    list_at_least_one_cstr = [
            'at_least_one_per_instance',
            'at_least_one_per_instance_unit_time',
            'fully_supervised',
            'spot_on',
            'at_least_one_per_temporal_point_unit_time',
            'at_least_one_per_instance_unit_time_with_keyframes',
            'at_least_one_per_temporal_point_unit_time_with_keyframes',
            'at_least_one_clip_level',
            'at_least_one_shot_level'
    ]
    if not need_init:
        at_least_one_solver = None
        pass
    elif cstrs_name in list_at_least_one_cstr:
        at_least_one_solver = linear_solvers.AtLeastOneSolver()
    else:
        raise ValueError(
            'cstrs_name: {} is not a valid option.'.format(cstrs_name))

    name_cstrs_init = 'cstrs_init_{}_{}_gb_{}_spth_{}_{}'.format(
        prepend_name, cstrs_name, group_by, sp_iou_thresh, mode)

    path_cstrs_init = os.path.join(cache_dir, name_cstrs_init + '.pkl')

    if not need_init:
        pass
    elif os.path.isfile(path_cstrs_init):
        with open(path_cstrs_init, 'rb') as f_cstrs:
            cstrs_init_dict = pickle.load(f_cstrs)
        asgn_train = cstrs_init_dict['asgn_train']
    else:
        n_rdm_init = 5
        for y, cstr in tqdm(zip(labels_train, cstrs_train)):
            # Generate a random gradient
            asgn_init = np.zeros(y.shape)

            for _ in range(n_rdm_init):
                rand_grad = np.random.randn(y.shape[0], y.shape[1])
                asgn_init += 1.0 / n_rdm_init * at_least_one_solver.solve(cstr, rand_grad)

            asgn_train.append(asgn_init)

        # Save to path.
        with open(path_cstrs_init, 'wb') as f_cstrs:
            pickle.dump({'asgn_train': asgn_train}, f_cstrs)

    # Create the experiment name.
    exp_name = '{}_{}_niter_{}_gb_{}_{}_spth_{}_lambda_{}_pratio_{}_{}_slack_{}_beta_{}_delta_{}{}'.format(
        prepend_name, cstrs_name, n_iterations, group_by, mode, sp_iou_thresh,
        alpha, -1, feat_type, False, -1.0, -1.0, exp_suffix)

    exp_name_val = re.sub('train_spth_', 'val_spth_', exp_name)

    # set out dirs/paths
    path_out = os.path.join(res_dir, exp_name)
    path_out_val = os.path.join(res_dir, exp_name_val)
    name_asgn = 'asgn.npy'.format(exp_name)
    path_asgn = os.path.join(path_out, name_asgn)

    # get a classifier
    name_w = 'w.pkl'.format(exp_name)
    path_w = os.path.join(path_out, name_w)

    call('mkdir -p {}'.format(path_out), shell=True)
   
    if video_eval_only:
        weights = None
        pass
    else:
        test_fun = None 
        if os.path.isfile(path_w):
            with open(path_w) as f_w:
                weights = pickle.load(f_w)
        else:
            # Launch FW optim.
            path_log_eval = os.path.join(path_log_eval, exp_name) if path_log_eval is not None else None
            print('Launching the BCFW optim...')
            asgn_final, weights = bcfw_diffrac.solver(
                feats_train,
                asgn_train,
                cstrs_train,
                at_least_one_solver,
                gts=labels_train,
                alpha=alpha,
                verbose='normal',
                bias_value=bias_value,
                block_sampling='gap_sampling',
                n_iterations=n_iterations,
                objective_frequency=250,
                eval_frequency=eval_frequency,
                eval_function=eval_functions.MAP(n_actions - 1, path_save=path_log_eval, save_pr_curves=save_pr_curves),
                validation_info=validation_info
                )
            # Save assignmnent.
            np.save(path_asgn, asgn_final)
            # save final classifier
            with open(path_w, 'wb') as f_w:
                pickle.dump(weights, f_w)

    if calibrate and not use_calibration:
        # need to test the validation videos to calibrate
        list_file = '{}/OF_vidlist_{}1.txt'.format(path_list, 'valtrainRate10_val')
    else:
        list_file = '{}/OF_vidlist_{}1.txt'.format(path_list, 'test')
    # Evaluate on test data.
    if video_eval_only:
        pass
    elif write_eval:
        print('Writing evaluation files in {}'.format(path_out))
        with open(list_file, 'r') as f_list:
            list_vid = f_list.readlines()
            list_vid = [x.strip().split(' ')[0] for x in list_vid]

        for id_vid, video_name in enumerate(tqdm(list_vid)):
            data_handler.write_eval_data(
                video_name,
                weights,
                bias_value=bias_value,
                path_root_out=os.path.join(path_out, 'tracks'),
                path_info=path_info,
                path_tracks=path_tracks,
                feat_type=feat_type,
                n_actions=n_actions,
                test_fun=test_fun)

    if write_eval or video_eval_only:
        if use_calibration:
            calib_path = os.path.join(path_out_val, 'calibration.pkl')
            print 'Load calibration from:\n{}'.format(calib_path)
            with open(calib_path) as f:
                calibration = pickle.load(f)
            loc_th = np.zeros((n_actions-1, len(video_eval_args['iou'])))
            for i, iou in enumerate(video_eval_args['iou']):
                loc_th[:, i] = calibration[iou]  # Set one th per action, per iou.
        else:
            loc_th = video_eval_args['loc_th']

        # clear eventual existing evaluation cache
        call('rm -rf  {}/evaluation_cache'.format(path_out), shell=True)

        # create instance to eval video mAP
        ev = evaluation.Evaluation(video_eval_args['datasetname'],
                                   [path_out], list_file,
                                   video_eval_args['iou'],
                                   smooth_window=25,
                                   loc_th=loc_th,
                                   track_class_agnostic=video_eval_args['track_class_agnostic'],
                                   force_no_regressor=True,
                                   nthreads=8,
                                   one_th_per_iou=use_calibration)
        return ev


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # Define default values for booleans.
    WRITE_EVAL = True

    # Parse args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', default='/sequoia/data2/jalayrac/nips2017weakpose/cache/')
    parser.add_argument('--res_dir', default='/sequoia/data2/jalayrac/nips2017weakpose/')
    parser.add_argument('--path_info', default='/sequoia/data2/gcheron/DALY/mytracksK5_50k')
    parser.add_argument('--path_tracks', default=
    '/sequoia/data2/gcheron/pytorch/diffrac_action_localization/DALY/results/mytracksK5_50k/tracks')
    parser.add_argument('--eval_frequency', type=int, default=500)
    parser.add_argument('--path_log_eval', type=str, default=None)
    parser.add_argument('--save_pr_curves', action='store_true', default=False)
    parser.add_argument('--no_init', action='store_true', default=False)
    parser.add_argument('--no_feat_init', action='store_true', default=False)
    parser.add_argument('--val_eval', action='store_true', default=False)
    parser.add_argument('--calibrate', action='store_true', default=False,
        help='learn/save the calibration parameters on the validation set')
    parser.add_argument('--use_calibration', action='store_true', default=False,
        help=('use the learnt calibration parameters, when combined with --calibrate, model trained on (train - val) '
              'is used, model trained on the whole train set is used otherwise'))
    parser.add_argument('--video_eval_only', action='store_true', default=False)
    parser.add_argument('--track_class_agnostic', action='store_true', default=False)
    parser.add_argument('--path_list', default='/sequoia/data2/gcheron/DALY')
    parser.add_argument('--n_iterations', type=int, default=0)
    parser.add_argument('--prepend_name', type=str, default='DALY_gtrack')
    parser.add_argument('--exp_suffix', type=str, default='')
    parser.add_argument('--datasetname', type=str, default='DALY')
    parser.add_argument(
        '--cstrs_name',
        type=str,
        default='at_least_one_per_instance_unit_time')
    parser.add_argument(
        '--write_eval',
        type=str2bool,
        nargs='?',
        const=True,
        default=WRITE_EVAL)
    parser.add_argument('--group_by', type=int, default=8)
    # UCF101: 25, DALY: 11
    parser.add_argument('--n_actions', type=int, default=11)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--rseed', type=int, default=19)
    args = parser.parse_args()

    # get eval params
    dname = args.datasetname

    if args.calibrate:
       tths = np.linspace(-0.2, 0.4, 13) #[-0.2:0.05:0.4]
       tths = np.concatenate( (np.array([-1e7]), tths)) # add -Inf

    else:
       tths = np.linspace(-0.1, 0.1, 11)

    if dname == 'DALY':
        loc_th = np.repeat(tths[None, :], 10, 0) # 10 x len(tths)
    elif dname == 'UCF101':
        if args.calibrate:
            loc_th = np.repeat(tths[None, :], 24, 0) # 24 x len(tths)
        else:
            loc_th = np.zeros((24,len(tths)))
            for i,tth in enumerate(tths):
                loc_th[:,i]=np.array([
                    tth,tth,-1e7,tth,tth,tth,-1e7,-1e7,tth,-1e7,-1e7,-1e7,-1e7,-1e7,tth,
                    -1e7,-1e7,-1e7,-1e7,tth,tth,-1e7,tth,-1e7])

    video_eval_args = {'datasetname': dname,
                       'iou': [ 0.2, 0.5],
                       'track_class_agnostic': args.track_class_agnostic,
                       'loc_th': loc_th}

    # Launch the job.
    print 'running with args:'
    print '================================='
    for field in args.__dict__.keys():
        print '{}: {}'.format(field, getattr(args, field))
    print '================================='
    ev = launcher(
        group_by=args.group_by,
        prepend_name=args.prepend_name,
        path_info=args.path_info,
        path_tracks=args.path_tracks,
        path_list=args.path_list,
        n_iterations=args.n_iterations,
        write_eval=args.write_eval,
        n_actions=args.n_actions,
        alpha=args.alpha,
        cache_dir=args.cache_dir,
        res_dir=args.res_dir,
        cstrs_name=args.cstrs_name,
        exp_suffix=args.exp_suffix,
        rdm_seed=args.rseed,
        eval_frequency=args.eval_frequency,
        path_log_eval=args.path_log_eval,
        save_pr_curves=args.save_pr_curves,
        val_eval=args.val_eval,
        video_eval_args=video_eval_args,
        video_eval_only=args.video_eval_only,
        calibrate=args.calibrate,
        use_calibration=args.use_calibration,
        no_feat_init=args.no_feat_init,
        no_init=args.no_init)

    # eval video mAP
    if ev is not None:
        calibration = ev.eval()
        if args.calibrate and not args.use_calibration:
            # not that if use_calibration, it would save calibration on test...
            calib_path = ev.trackpath[0] + '/../calibration.pkl'
            assert not os.path.exists(calib_path), ('calibration already exists!\n{}'.format(calib_path))
            print 'writting calibration in :\n{}'.format(calib_path)
            with open(calib_path, 'wb') as f:
                pickle.dump(calibration, f)
