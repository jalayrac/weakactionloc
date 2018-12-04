import pickle
import numpy as np
import os
import glob
import re
import track_utils
import scipy.io as sio


def is_inside(interval, reference, thresh=0.5):
    intersection = (max(interval[0], reference[0]),
                    min(interval[1], reference[1]))

    return intersection[1] - intersection[0] + 1 > thresh * (
        interval[1] - interval[0] + 1)


def get_cstr_for_video(
        video_name,
        path_info='/sequoia/data2/gcheron/UCF101/detection/mytracksK5_600k',
        path_tracks='/sequoia/data2/gcheron/pytorch/diffrac_action_localization/UCF101/results/mytracksK5_600k/tracks/',
        groupby=8,
        n_actions=25,
        constr_fn=None,
        ignore_ids=[]):
    """ Get data for individual video.
    Args:
        video_name:
        path_info:
        path_tracks:
        groupby: groupby this number of frames.
        n_actions:
        constr_fn: function to call to get constraints
        ignore_ids: eventually ignore these tracklet ids
    Returns:
        x: list of features.
        z: list of ground-truth.
    """

    info_video = pickle.load(
        open(os.path.join(path_info, video_name + '.pkl')))
    gt = info_video['gts']

    if not gt:
        return False, 0

    # Read the tracks.
    list_track_file = sorted(
        glob.glob(os.path.join(path_tracks, video_name, '*pkl')))

    # We create tmp_cstr as follows:
    # tmp_cstr['background'] = list of tracklet id belonging to background.
    # tmp_cstr['tracklet_id'] = list of candidate action ids for that tracklet.
    # tmp_cstr['id_action']['idx_gt'] = list of tracklet that intersect with that instance of GT.
    # tmp_cstr['class'] = list of tracklet for which we know the exact class.

    tmp_cstr = {'background': [], 'classes': {}}

    # id of currently added tracklets
    id_tracklet = 0
    # id of tracklets from the whole set
    id_tracklet_global = 0

    extra_info = {'video_length': info_video['length'], 'n_actions': n_actions, 'video_name': video_name,
                  'path_info': path_info, 'gt_display': []}
    for id_track, track_name in enumerate(list_track_file):
        track_data = pickle.load(open(track_name, 'rb'))
        n_chunks = (track_data['N_frames'] - 1) / groupby + 1
        # Start/end of track (start at 1).
        start_frame = track_data['tbound'][0]
        end_frame = track_data['tbound'][1]

        for t in range(n_chunks):
            if id_tracklet_global in ignore_ids:
                id_tracklet_global += 1
                continue

            # Get the start/end of the tracklet in the original video to be comparable with GT.
            start_chunk = t * groupby + start_frame
            end_chunk = min(start_chunk + groupby, end_frame + 1)
            tbound_track = (start_chunk, end_chunk - 1)

            extra_info['track_info'] = info_video['tracks'][id_track]

            # call the desired contraint function and set tmp_cstr inside
            has_constraint, key_tracklet = constr_fn(id_tracklet, tbound_track,
                                                     groupby, gt, tmp_cstr,
                                                     extra_info)

            if not has_constraint:
                # If the tracklet was not matching any GT, force it to background.
                # Note: this won't be use if key_tracklet is in tmp_cstr['classes']
                tmp_cstr['background'].append(id_tracklet)
            else:
                # Otherwise, add background as a potential candidate.
                # Note: this won't be use if key_tracklet is in tmp_cstr['classes']
                tmp_cstr[key_tracklet].add(n_actions - 1)

            # Increment the id of the tracklet.
            id_tracklet_global += 1
            id_tracklet += 1

    cstr = {'equal_0': [], 'equal_1': [], 'row_eq_1': [], 'col_geq_1': []}
    n_tracklet = id_tracklet

    # If the key_tracklet is in tmp_cstr['classes'], it will directly force all classes to be either 0 or 1
    # Hence, tmp_cstr['background'], tmp_cstr[key_tracklet] will not be used and no cstr['row_eq_1'] will be set.
    # However, cstr['col_geq_1'] will still be set.
    for i in range(n_tracklet): 
        tkey = 'tracklet_{}'.format(i)
        if tkey in tmp_cstr['classes']:
            # We know exact class for this tracklet
            c_class = tmp_cstr['classes'][tkey]
            if c_class == -1:
                # this is background
                c_class = n_actions - 1
            for k in range(n_actions):
                _pos = n_actions * i + k
                if k == c_class:
                    cstr['equal_1'].append(_pos)
                else:
                    cstr['equal_0'].append(_pos)

        else:
            # Except if background, exact class is unknown
            # Check whether or not the tracklet belongs to background.
            if i in tmp_cstr['background']:
                # background to 1
                cstr['equal_1'].append(n_actions * i + n_actions - 1)
                for k in range(n_actions - 1):
                    # all other actions to 0
                    cstr['equal_0'].append(n_actions * i + k)
            else:
                cur_row_array = []
                for k in range(n_actions):
                    if k in tmp_cstr[tkey]:
                        # this tracklet can be action k
                        cur_row_array.append(n_actions * i + k)
                    else:
                        # If that tracklet does not match that action, force it to 0.
                        cstr['equal_0'].append(n_actions * i + k)
                # row must sum to 1 on all potential actions (including background)
                cstr['row_eq_1'].append(np.array(cur_row_array))

    # Loop over actions.
    for k in range(n_actions - 1):
        if k in tmp_cstr:
            # Loop over instances.
            for bag_k in tmp_cstr[k]:
                cstr['col_geq_1'].append(
                    [n_actions * i + k for i in tmp_cstr[k][bag_k]])

    return True, cstr


def get_at_least_one_per_instance_unit_time_with_keyframes_cstr_for_video(
        id_tracklet, tbound_track, groupby, gt, tmp_cstr, extra_info):
    """ Get data for individual video.
    Args:
        id_tracklet:
        tbound_track:
        groupby:
        gt:
        tmp_cstr:
        extra_info: Not used here
    Returns:
        has_constraint:
        key_tracklet:
    """
    key_tracklet = 'tracklet_{}'.format(id_tracklet)

    # Spatial overlap with keyframes
    sp_overlap_keyframes = extra_info['track_info']['ann_iou']
    ov_threshold = 0.3
    # Check if it interacts with GT.
    has_constraint = False
    max_overlap = 0.0
    for idx_gt, i_gt in enumerate(gt):
        if is_inside(tbound_track, i_gt['tbound']):
            # Three cases can happen when the tracklet is in the GT interval:
            # a) The track of that tracklet overlap enough with the keyframes -> force to that action
            # b) The track of that tracklet does not overlap enough with the keyframes -> don't consider it as a
            # potential candidate
            # c) The track of that tracklet does not have overlap in time with the keyframes -> consider it as a
            # potential candidate for that action

            all_nan = np.all(np.isnan(sp_overlap_keyframes[:, idx_gt]))

            id_action = i_gt['label'] - 1

            if all_nan or np.nanmin(sp_overlap_keyframes[:, idx_gt]) > ov_threshold:
                # Case c) or Case a)
                t_representer = ((tbound_track[0] + tbound_track[1]) / 2) / groupby
                key_bag = '{}_{}'.format(idx_gt, t_representer)
                has_constraint = True
                if key_tracklet not in tmp_cstr:
                    tmp_cstr[key_tracklet] = set()

                if all_nan:
                    # Case c) -> This is a candidate action for this tracklet
                    if key_tracklet not in tmp_cstr['classes']:
                        tmp_cstr[key_tracklet].add(id_action)
                    else:
                        print('{} ({}): attempt to add as potential candidate a tracklet that was already force to'
                              ' a GT, cleaning that for you...'.format(extra_info['video_name'], key_tracklet))
                        continue
                else:
                    # Case a) -> Force that class
                    if key_tracklet in tmp_cstr['classes']:
                        if np.nanmin(sp_overlap_keyframes[:, idx_gt]) > max_overlap:
                            max_overlap = np.nanmin(sp_overlap_keyframes[:, idx_gt])
                            print('{} ({}): you changed the forced id_action of a tracklet (bigger ov)!'.format(
                                extra_info['video_name'], key_tracklet))
                        else:
                            print('{} ({}): you attempted to change a forced id_action (smaller ov)!'.format(
                                extra_info['video_name'], key_tracklet))
                            continue

                    tmp_cstr['classes'][key_tracklet] = id_action
                    for other_action in range(extra_info['n_actions']):
                        if other_action in tmp_cstr[key_tracklet]:
                            print('{}: forcing an action that already matched another action, cleaning...'.format(
                                extra_info['video_name']))
                            tmp_cstr[key_tracklet].remove(other_action)

                        if other_action != id_action:
                            if other_action in tmp_cstr:
                                if key_bag in tmp_cstr[other_action]:
                                    if id_tracklet in tmp_cstr[other_action][key_bag]:
                                        print('{}: forcing an action that already matched another action, cleaning...'.format(
                                            extra_info['video_name']))
                                        tmp_cstr[other_action][key_bag].remove(id_tracklet)

                if id_action not in tmp_cstr:
                    tmp_cstr[id_action] = {}

                if key_bag not in tmp_cstr[id_action]:
                    tmp_cstr[id_action][key_bag] = []

                # this tracklet is a candidate to match this gt at this time
                # (at least one tracklet will match the GT at this particular time)
                tmp_cstr[id_action][key_bag].append(id_tracklet)
            else:
                # Case b) -> just ignore it
                pass
    return has_constraint, key_tracklet


def get_at_least_one_clip_level_cstr_for_video(
        id_tracklet, tbound_track, groupby, gt, tmp_cstr, extra_info):
    """ Get data for individual video.
    Args:
        id_tracklet:
        tbound_track:
        groupby:
        gt:
        tmp_cstr:
        extra_info: Not used here
    Returns:
        has_constraint:
        key_tracklet:
    """
    key_tracklet = 'tracklet_{}'.format(id_tracklet)

    # Every GT label are considered as a positive
    has_constraint = True
    for idx_gt, i_gt in enumerate(gt):
        id_action = i_gt['label'] - 1
        key_bag = '{}_{}'.format(idx_gt, id_action)

        if key_tracklet not in tmp_cstr:
            tmp_cstr[key_tracklet] = set()

        # this is a candidate action for this tracklet
        tmp_cstr[key_tracklet].add(id_action)

        if id_action not in tmp_cstr:
            tmp_cstr[id_action] = {}

        if key_bag not in tmp_cstr[id_action]:
            tmp_cstr[id_action][key_bag] = []

        # this tracklet is a candidate to match this gt 
        # (at least one tracklet will match in this GT video)
        tmp_cstr[id_action][key_bag].append(id_tracklet)

    return has_constraint, key_tracklet


def get_at_least_one_shot_level_cstr_for_video(
        id_tracklet, tbound_track, groupby, gt, tmp_cstr, extra_info):
    """ Get data for individual video.
    Args:
        id_tracklet:
        tbound_track:
        groupby:
        gt:
        tmp_cstr:
        extra_info: use path_info, video_name, video_length, gt_display
    Returns:
        has_constraint:
        key_tracklet:
    """
    key_tracklet = 'tracklet_{}'.format(id_tracklet)

    def get_corrected_shot(shot_path, vname):
        _shots = sio.loadmat(shot_path + '/' + vname + '.mp4.mat')['shots']

        # fix the shots
        _shots -= 1
        _shots[0,0] = 1
        _shots[-1,1] = extra_info['video_length']

        assert (_shots <= _shots[-1,1]).all(), 'some previous shots goes after the video length'
        
        return _shots.astype(int)

    vidname = extra_info['video_name']
    shots = get_corrected_shot(extra_info['path_info'] + '/../shots', vidname)

    def get_shot(tbound, shots, id_GT, extra_info):
        # return the shot with the largest intersection
        vidname = extra_info['video_name']
        isGT = id_GT >= 0

        inters = np.zeros(len(shots))
        for i_shot, shot in enumerate(shots):
            t1_max = max(tbound[0], shot[0])
            t2_min = min(tbound[1], shot[1])
            if t1_max > t2_min:
               # no intersection
               continue
            inters[i_shot] = t2_min - t1_max + 1 

        if (inters <= 0).all():
            print '\n\nERROR: no intersection with any shots\ncshot:\n', tbound, '\nshot list:\n', shots
            raise ValueError

        if (inters > 0).sum() > 1:
            if not isGT:
                print '{}: Tracklet ({}, {}) is candidate for several shots with intersion:'.format(
                       vidname, tbound[0], tbound[1])
                print inters

            if (inters > 5).sum() > 1:
                # we allow only a small overlap with several shots
                # OTHERWISE: annotation contains errors, check which shots we add
                assert isGT, 'for tracklets we are suppoed to find only one shot'
                display_key = '{}_{}'.format(vidname, id_GT) # display only once per video/per gt instance

                display_ = False
                if display_key not in extra_info['gt_display']:
                    display_ = True
                    extra_info['gt_display'].append(display_key)

                    print '{}: GT ({}, {}) is candidate for several shots with intersion:'.format(
                           vidname, tbound[0], tbound[1])
                    print inters

                shot_list = []

                # select all shots for which the intersection is > 5% of the GT length
                tlen = tbound[1] - tbound[0] + 1
                shot_list = np.argwhere((inters/tlen) > 0.05 ).flatten().tolist()
               
                if len(shot_list) == 0:
                    shot_list = [ (inters).argmax() ]

                if display_:
                    print 'pick shot(s)/of intersections:'
                    print shot_list
                    print inters[shot_list]

                return shot_list

        if isGT:
            # the GT has one shot in its list
            return [ (inters).argmax() ]

        # this is a tracklet, return the best shot candidate
        return (inters).argmax() 

    # get the tracklet shot
    tracklet_shot = get_shot(tbound_track, shots, -1, extra_info)

    # get gt shots
    gt_shots = []
    for idx_gt, i_gt in enumerate(gt):
        gt_shots.append(get_shot(i_gt['tbound'], shots, idx_gt, extra_info))

    has_constraint = False
    for idx_gt, i_gt in enumerate(gt):

        if tracklet_shot in gt_shots[idx_gt]:
            # if the GT is in the tracklet shot, consider its label
            has_constraint = True

            id_action = i_gt['label'] - 1
            key_bag = '{}_{}'.format(tracklet_shot, id_action)

            if key_tracklet not in tmp_cstr:
                tmp_cstr[key_tracklet] = set()

            # this is a candidate action for this tracklet
            tmp_cstr[key_tracklet].add(id_action)

            if id_action not in tmp_cstr:
                tmp_cstr[id_action] = {}

            if key_bag not in tmp_cstr[id_action]:
                tmp_cstr[id_action][key_bag] = []

            # this tracklet is a candidate to match this gt 
            # (at least one tracklet will match this GT in this shot)
            if id_tracklet not in tmp_cstr[id_action][key_bag]:
                # if the id has been added by another GT of the same class in the same shot
                tmp_cstr[id_action][key_bag].append(id_tracklet)

    return has_constraint, key_tracklet


def get_at_least_one_per_instance_unit_time_cstr_for_video(
        id_tracklet, tbound_track, groupby, gt, tmp_cstr, extra_info):
    """ Get data for individual video.
    Args:
        id_tracklet:
        tbound_track:
        groupby:
        gt:
        tmp_cstr:
        extra_info: Not used here
    Returns:
        has_constraint:
        key_tracklet:
    """
    key_tracklet = 'tracklet_{}'.format(id_tracklet)

    # Check if it interacts with GT.
    has_constraint = False
    for idx_gt, i_gt in enumerate(gt):
        if is_inside(tbound_track, i_gt['tbound']):

            t_representer = ((tbound_track[0] + tbound_track[1]) / 2) / groupby

            id_action = i_gt['label'] - 1
            if key_tracklet not in tmp_cstr:
                tmp_cstr[key_tracklet] = set()

            # this is a candidate action for this tracklet
            tmp_cstr[key_tracklet].add(id_action)
            has_constraint = True

            key_bag = '{}_{}'.format(idx_gt, t_representer)

            if id_action not in tmp_cstr:
                tmp_cstr[id_action] = {}

            if key_bag not in tmp_cstr[id_action]:
                tmp_cstr[id_action][key_bag] = []

            # this tracklet is a candidate to match this gt at this time
            # (at least one tracklet will match the GT at this particular time)
            tmp_cstr[id_action][key_bag].append(id_tracklet)

    return has_constraint, key_tracklet


def get_at_least_one_per_temporal_point_unit_time_with_keyframes_cstr_for_video(
        id_tracklet, tbound_track, groupby, gt, tmp_cstr, extra_info):
    """ This constraint takes the middle point of a GT interval and
        considers action happens during a fix interval center at this point
    Args:
        id_tracklet:
        tbound_track:
        groupby:
        gt:
        tmp_cstr:
        extra_info: use 'video_length'
    Returns:
        has_constraint:
        key_tracklet:
    """
    fixed_delta = 25  # Fixed delta frames before and after the middle point will be consider as GT

    key_tracklet = 'tracklet_{}'.format(id_tracklet)

    # Spatial overlap with keyframe
    sp_overlap_keyframes = extra_info['track_info']['ann_iou']
    ov_threshold = 0.3

    has_constraint = False
    max_overlap = 0.0
    # Check if it interacts with GT.
    for idx_gt, i_gt in enumerate(gt):
        all_nan = np.all(np.isnan(sp_overlap_keyframes[:, idx_gt]))
        # Possible cases:
        # a) Inside delta interval, ov > thresh -> force to action
        # b) Inside delta interval, ov < thresh -> force to background
        # c) Inside delta interval, ov allnan -> potential positive (included in the alo)
        # d) Outside delta interval, ov > thresh -> potential positive (not included in the alo)
        # e) Outside delta interval, ov < thresh -> force to background
        # f) Outside delta interval, ov allnan -> potential positive (not included in the alo)

        if all_nan or np.nanmin(sp_overlap_keyframes[:, idx_gt]) > ov_threshold:
            assert len(i_gt['used_ann']) == 1, 'Constraint not supported with more than 1 keyframe per GT instance'
            keyframe_middle_point = int((i_gt['used_ann'][0]['tbound']))
            t1 = max(1, keyframe_middle_point - fixed_delta)
            t2 = min(extra_info['video_length'], keyframe_middle_point + fixed_delta)
            interval_tbound = (t1, t2)

            has_constraint = True
            id_action = i_gt['label'] - 1

            if key_tracklet not in tmp_cstr:
                tmp_cstr[key_tracklet] = set()

            if is_inside(tbound_track, interval_tbound):
                t_representer = ((tbound_track[0] + tbound_track[1]) / 2) / groupby
                key_bag = '{}_{}'.format(idx_gt, t_representer)
                # Case a) or Case c)
                if all_nan:
                    # Case c) -> This is a candidate action for this tracklet
                    if key_tracklet not in tmp_cstr['classes']:
                        tmp_cstr[key_tracklet].add(id_action)
                    else:
                        print('{} ({}): attempt to add as potential candidate a tracklet that was already force to'
                              ' a GT, cleaning that for you...'.format(extra_info['video_name'], key_tracklet))
                        continue
                else:
                    # Case a) -> Force that class
                    if key_tracklet in tmp_cstr['classes']:
                        if np.nanmin(sp_overlap_keyframes[:, idx_gt]) > max_overlap:
                            max_overlap = np.nanmin(sp_overlap_keyframes[:, idx_gt])
                            print('{} ({}): you changed the forced id_action of a tracklet (k-ov >)!'.format(
                                extra_info['video_name'], key_tracklet))
                        else:
                            print('{} ({}): you attempted to change a forced id_action (but its k-ov was <)!'.format(
                                extra_info['video_name'], key_tracklet))
                            continue

                    tmp_cstr['classes'][key_tracklet] = id_action
                    for other_action in range(extra_info['n_actions']):
                        if other_action in tmp_cstr[key_tracklet]:
                            print('{}: forcing an action that already matched another action, cleaning...'.format(
                                extra_info['video_name']))
                            tmp_cstr[key_tracklet].remove(other_action)

                        if other_action != id_action:
                            if other_action in tmp_cstr:
                                if key_bag in tmp_cstr[other_action]:
                                    if id_tracklet in tmp_cstr[other_action][key_bag]:
                                        print('{}: forcing an action that already matched another action, cleaning...'.format(
                                            extra_info['video_name']))
                                        tmp_cstr[other_action][key_bag].remove(id_tracklet)

                # Add this tracklet to the bag for the alo constraint
                if id_action not in tmp_cstr:
                    tmp_cstr[id_action] = {}

                if key_bag not in tmp_cstr[id_action]:
                    tmp_cstr[id_action][key_bag] = []

                # This tracklet is a candidate to match this gt at this time
                # (at least one tracklet will match the GT at this particular time)
                tmp_cstr[id_action][key_bag].append(id_tracklet)
            else:
                # Case d) or Case e) -> add as a potential candidate without imposing the alo constraint
                if key_tracklet not in tmp_cstr['classes']:
                    tmp_cstr[key_tracklet].add(id_action)
                else:
                    print('{}: forcing an action that already matched another action, cleaning...'.format(extra_info['video_name']))
                    continue
        else:
            # case b) or case e) -> just ignore it
            pass

    return has_constraint, key_tracklet


def get_at_least_one_per_temporal_point_unit_time_cstr_for_video(
            id_tracklet, tbound_track, groupby, gt, tmp_cstr, extra_info):
    """ This constraint takes the middle point of a GT interval and
        considers action happens during a fix interval center at this point
    Args:
        id_tracklet:
        tbound_track:
        groupby:
        gt:
        tmp_cstr:
        extra_info: use 'video_length'
    Returns:
        has_constraint:
        key_tracklet:
    """
    fixed_delta = 25 # fixed delta frames before and after the middle point will be consider as GT
    
    key_tracklet = 'tracklet_{}'.format(id_tracklet)

    has_one_cstr = False
    action_labels = [ i_gt['label'] - 1 for i_gt in gt ]
    # Check if it interacts with GT.
    for idx_gt, i_gt in enumerate(gt):
        #keyframe_temporal_point = int( (i_gt['tbound'][0] + i_gt['tbound'][1]) / 2 ) # get the gt middle point

        # get the same temporal point as the one used for the "keyframe" contraint
        assert len(i_gt['used_ann']) == 1, 'Constraint not supported with more than 1 keyframe per GT instance'
        keyframe_temporal_point = int((i_gt['used_ann'][0]['tbound']))

        t1 = max(1, keyframe_temporal_point - fixed_delta)
        t2 = min(extra_info['video_length'], keyframe_temporal_point + fixed_delta)
        interval_tbound = (t1, t2) 
        if is_inside(tbound_track, interval_tbound):
            has_one_cstr = True

            t_representer = ((tbound_track[0] + tbound_track[1]) / 2) / groupby

            id_action = i_gt['label'] - 1
            if key_tracklet not in tmp_cstr:
                tmp_cstr[key_tracklet] = set()

            # this is a candidate action for this tracklet
            tmp_cstr[key_tracklet].add(id_action)

            key_bag = '{}_{}'.format(idx_gt, t_representer)

            if id_action not in tmp_cstr:
                tmp_cstr[id_action] = {}

            if key_bag not in tmp_cstr[id_action]:
                tmp_cstr[id_action][key_bag] = []

            # this tracklet is a candidate to match this gt at this time
            # (at least one tracklet will match the GT at this particular time)
            tmp_cstr[id_action][key_bag].append(id_tracklet)

    if not has_one_cstr:
        # we have no information except the action labels in the video
        # bkg is added after this fun
        assert not key_tracklet in tmp_cstr
        tmp_cstr[key_tracklet] = set(action_labels)

    has_constraint = True # all tracklets have constraint here
    return has_constraint, key_tracklet


def get_at_least_one_per_instance_cstr_for_video(id_tracklet, tbound_track,
                                                 groupby, gt, tmp_cstr, extra_info):
    """ Get data for individual video.
    Args:
        id_tracklet:
        tbound_track:
        groupby:
        gt:
        tmp_cstr:
        extra_info: Not used here
    Returns:
        has_constraint:
        key_tracklet:
    """
    key_tracklet = 'tracklet_{}'.format(id_tracklet)

    # Check if it interacts with GT.
    has_constraint = False
    for idx_gt, i_gt in enumerate(gt):
        if is_inside(tbound_track, i_gt['tbound']):
            id_action = i_gt['label'] - 1
            if key_tracklet not in tmp_cstr:
                tmp_cstr[key_tracklet] = set()

            # this is a candidate action for this tracklet
            tmp_cstr[key_tracklet].add(id_action)
            has_constraint = True

            if id_action not in tmp_cstr:
                tmp_cstr[id_action] = {}

            if idx_gt not in tmp_cstr[id_action]:
                tmp_cstr[id_action][idx_gt] = []

            # this tracklet is a candidate to match this gt (at least one tracklet will match the whole GT)
            tmp_cstr[id_action][idx_gt].append(id_tracklet)

    return has_constraint, key_tracklet


def get_fully_supervised_cstr_for_video(id_tracklet, tbound_track, groupby, gt, tmp_cstr, extra_info):
    """ Get data for individual video.
    Args:
        id_tracklet:
        tbound_track:
        groupby:
        gt:
        tmp_cstr:
        extra_info: use 'track_info'
    Returns:
        has_constraint:
        key_tracklet:
    """
    key_tracklet = 'tracklet_{}'.format(id_tracklet)
    if key_tracklet not in tmp_cstr:
        # note this field will remain empty here (since we know the exact class)
        tmp_cstr[key_tracklet] = set()

    # in tmp_cstr, only set ['classes'] since we are in fully supervised setup 
    positive_th = 0.3

    t_info = extra_info['track_info']
    t1, t2, _ = track_utils.framebounds2idx(t_info, tbound_track)
    gt_chunk = np.min(
        t_info['gt_iou'][t1:t2+1, :], axis=0)

    id_max = np.argmax(gt_chunk)
    max_iou = gt_chunk[id_max]

    if max_iou >= positive_th:
        _class = gt[id_max]['label'] - 1
    else:
        _class = -1 # backgroud 

    assert not key_tracklet in tmp_cstr['classes']
    tmp_cstr['classes'][key_tracklet] = _class

    has_constraint = True
    return has_constraint, key_tracklet


def get_spot_on_cstr_for_video(id_tracklet, tbound_track, groupby, gt, tmp_cstr, extra_info):
    """ Get data for individual video.
    Args:
        id_tracklet:
        tbound_track:
        groupby:
        gt:
        tmp_cstr:
        extra_info: use 'track_info'
    Returns:
        has_constraint:
        key_tracklet:
    """
    key_tracklet = 'tracklet_{}'.format(id_tracklet)
    if key_tracklet not in tmp_cstr:
        # note this field will remain empty here (since we know the exact class)
        tmp_cstr[key_tracklet] = set()

    # in tmp_cstr, only set ['classes'] since we are in a fully supervised type of setup
    t_info = extra_info['track_info']
    t1, t2, _ = track_utils.framebounds2idx(t_info, tbound_track)

    gts_boxes = t_info['gt_boxes'][t1:t2+1, :, :]
    track_boxes = t_info['boxes'][t1:t2+1, :]

    dist_to_center = 100000 * np.ones(gts_boxes.shape[1])
    for i_gt in range(gts_boxes.shape[1]):
        gt_boxes = gts_boxes[:, i_gt, :]
        if np.isnan(gt_boxes).any():
            continue

        # Check if all spots are inside the track boxes
        spot_inside = True
        for gt_box, track_box in zip(gt_boxes, track_boxes):
            center_gt = track_utils.get_center_box(gt_box)
            if not track_utils.spot_inside_box(center_gt, track_box):
                spot_inside = False
                break

            # If inside, record the distance to the center in order to choose the best instance
            center_track = track_utils.get_center_box(track_box)
            dist_to_center[i_gt] = min(dist_to_center[i_gt], track_utils.dist_spots(center_track, center_gt))

        if not spot_inside:
            dist_to_center[i_gt] = 100000

    # Select the action instance with the smallest distance to center
    id_min = np.argmin(dist_to_center)
    min_dist = dist_to_center[id_min]

    if min_dist < 1000:
        _class = gt[id_min]['label'] - 1
    else:
        _class = -1  # background

    assert not key_tracklet in tmp_cstr['classes']
    tmp_cstr['classes'][key_tracklet] = _class

    has_constraint = True
    return has_constraint, key_tracklet


def get_data_for_video(
        video_name,
        path_info='/sequoia/data2/gcheron/UCF101/detection/mytracksK5_600k',
        path_tracks='/sequoia/data2/gcheron/pytorch/diffrac_action_localization/UCF101/results/mytracksK5_600k/tracks/',
        sp_iou_thresh=0.3,
        groupby=8,
        feat_type='RGB+OPF',
        dim_feat=832,
        n_actions=25):
    """ Get data for individual video.
    Args:
        video_name:
        path_info:
        path_tracks:
        sp_iou_thresh:
        groupby: grouby this number of frames
        feat_type: use + if you want to combine features.
        dim_feat: dimension of individual feature.
        n_actions:
    Returns:
        x: list of features.
        z: list of ground-truth.
    """

    info_video = pickle.load(
        open(os.path.join(path_info, video_name + '.pkl')))
    gt = info_video['gts']

    if not gt:
        return False, 0, 0

    labels = np.array([x['label'] - 1 for x in gt])

    split_feat = feat_type.split('+')
    n_feat_type = len(split_feat)

    # Read the tracks.
    list_track_file = sorted(
        glob.glob(os.path.join(path_tracks, video_name, '*pkl')))

    if len(list_track_file) == 0:
        return False, 0, 0

    feats = []
    gts = []
    for id_track, track_name in enumerate(list_track_file):
        assert re.match('.*/([^/]*)', track_name).group(1) == 'track{:05d}.pkl'.format(id_track + 1)
        track_data = pickle.load(open(track_name, 'rb'))
        n_chunks = (track_data['N_frames'] - 1) / groupby + 1
        start_frame = track_data['tbound'][0] - 1

        x = np.zeros([n_chunks, dim_feat * n_feat_type])
        z = np.zeros([n_chunks, n_actions])

        for t in range(n_chunks):
            start_chunk = t * groupby
            end_chunk = min(start_chunk + groupby, track_data['N_frames'])

            # Deduce the GT for that chunk.
            gt_chunk = np.min(
                info_video['tracks'][id_track]['gt_iou'][start_chunk:end_chunk, :], axis=0)
            id_max = np.argmax(gt_chunk)

            # Add the GT to z.
            if sp_iou_thresh > 0:
                # Create 0/1 labels.
                if gt_chunk[id_max] > sp_iou_thresh:
                    z[t, labels[id_max]] = 1
                else:
                    z[t, -1] = 1
            else:
                # Regress the IOU instead of creating 0/1 labels.
                if gt_chunk[id_max] > -sp_iou_thresh:
                    # If higher than threshold, put iou as regressed value.
                    z[t, labels[id_max]] = gt_chunk[id_max]
                else:
                    # If lower than threshold, put 1-iou as regressed value for background.
                    z[t, -1] = 1 - gt_chunk[id_max]

            # Get the feature for that chunk.
            for f_id, feat_type in enumerate(split_feat):
                feat_chunk = np.zeros(dim_feat)
                for frame in range(start_chunk, end_chunk):
                    # Get the original chunk id from feature computation (not compatible with Python 3).
                    id_feat = (frame + start_frame) / 4 - start_frame / 4
                    # Get the feature and divide by the total number of frames in my track chunk.
                    feat_chunk += track_data[feat_type][id_feat, :] / (
                        end_chunk - start_chunk)
                # Fill x with the average feature inside the chunk.
                x[t, f_id * dim_feat:(f_id + 1) * dim_feat] = feat_chunk

        feats.append(x)
        gts.append(z)

    feats = np.concatenate(feats)
    gts = np.concatenate(gts)
    return True, feats, gts


def write_eval_data(
        video_name,
        W,
        bias_value=100,
        path_root_out='/sequoia/data2/jalayrac/nips2017weakpose/UCF101',
        path_info='/sequoia/data2/gcheron/UCF101/detection/mytracksK5_600k',
        path_tracks='/sequoia/data2/gcheron/pytorch/diffrac_action_localization/UCF101/results/mytracksK5_600k/tracks/',
        feat_type='RGB',
        dim_feat=832,
        n_actions=25,
        test_fun=None):
    """ Get data for individual video.
    Args:
        video_name:
        path_info:
        path_tracks:
        iou_thresh:
        groupby: grouby this number of frames
        feat_type: use + if you want to combine features.
        dim_feat: dimension of individual feature.
        n_actions:
        test_fun: if None just do X*W
    Returns:
        x:
        z:
    """

    info_video = pickle.load(
        open(os.path.join(path_info, video_name + '.pkl')))
    gt = info_video['gts']

    if not gt:
        return False

    labels = np.array([x['label'] - 1 for x in gt])

    split_feat = feat_type.split('+')
    n_feat_type = len(split_feat)

    # Read the tracks.
    list_track_file = sorted(
        glob.glob(os.path.join(path_tracks, video_name, '*pkl')))

    # print('Dealing with video {}'.format(video_name))
    for id_track, track_name in enumerate(list_track_file):
        assert re.match('.*/([^/]*)', track_name).group(1) == 'track{:05d}.pkl'.format(id_track + 1)

        # track_data_feat: contains features
        track_data_feat = pickle.load(open(track_name, 'rb'))
        n_chunks = track_data_feat['RGB'].shape[0]
        start_frame = track_data_feat['tbound'][0] - 1
        x = np.zeros([n_chunks, dim_feat * n_feat_type])

        for f_id, feat_type in enumerate(split_feat):
            x[:, f_id * dim_feat:(f_id + 1) * dim_feat] = track_data_feat[feat_type]

        if test_fun is not None:
           scores = test_fun(x, W)
        else:
           x_w_bias = np.append(x, bias_value * np.ones([x.shape[0], 1]), axis=1)
           scores = np.dot(x_w_bias, W)

        # track_data: contains track info for saving
        track_data = info_video['tracks'][id_track]
        assert (
            track_data['boxes'].shape[0] == track_data_feat['boxes'].shape[0]) and (
            track_data['tbound'] == track_data_feat['tbound']) and (
            n_chunks == track_data['frame2chunk'][-1] - track_data['frame2chunk'][0] + 1, (
            'track files do not correspond'))

        track_data['scores'] = scores

        # Store results.
        dir_out = os.path.join(path_root_out, video_name)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        with open(os.path.join(dir_out, os.path.basename(track_name)),
                  'wb') as f_out:
            pickle.dump(track_data, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    return True


def get_general_cstr_for_video(
        video_name,
        path_info='/sequoia/data2/gcheron/UCF101/detection/mytracksK5_600k',
        path_tracks='/sequoia/data2/gcheron/pytorch/diffrac_action_localization/UCF101/results/mytracksK5_600k/tracks/',
        groupby=8,
        n_actions=25,
        constr_fn_name=None,
        ignore_ids=[]):

    if constr_fn_name in [
            'get_at_least_one_per_instance_cstr_for_video',
            'get_at_least_one_per_instance_unit_time_cstr_for_video',
            'get_fully_supervised_cstr_for_video',
            'get_spot_on_cstr_for_video',
            'get_at_least_one_per_temporal_point_unit_time_cstr_for_video',
            'get_at_least_one_per_temporal_point_unit_time_with_keyframes_cstr_for_video',
            'get_at_least_one_per_instance_unit_time_with_keyframes_cstr_for_video',
            'get_at_least_one_clip_level_cstr_for_video',
            'get_at_least_one_shot_level_cstr_for_video'
    ]:
        present, cstr = get_cstr_for_video(
            video_name=video_name,
            path_info=path_info,
            path_tracks=path_tracks,
            groupby=groupby,
            n_actions=n_actions,
            constr_fn=globals()[constr_fn_name],
            ignore_ids=ignore_ids)
    else:
        raise ValueError('{} not defined.'.format(constr_fn_name))

    return present, cstr
