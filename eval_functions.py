import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import os
import pickle


class Evaluator:
    def __init__(self, path_save=None):
        self.path_save = path_save
        if self.path_save is not None:
            if not os.path.exists(self.path_save):
                os.makedirs(self.path_save)
        self.call_number = 0

    def evaluate(self, asgns, gts, weights, feats, rounding_function, cstrs):
        raise NotImplementedError


class MAP(Evaluator):
    """Perform MAP evaluation.

    NB: it assumes that the features are stored in RAM.

    """

    def __init__(self, n_actions, path_save=None, save_pr_curves=False):
        Evaluator.__init__(self, path_save)
        self.save_pr_curves = save_pr_curves
        self.n_actions = n_actions # should not contain bkg

    def evaluate(self, asgns, gts, weights, feats, rounding_function, ocstrs):
        n_actions = self.n_actions
        if asgns is None or ocstrs is None:
            isVal = True
        else:
            isVal = False
            assert n_actions == asgns[0].shape[1] - 1
        self.call_number += 1

        if not isVal and 'group_tracklet' in ocstrs[0]:
            cstrs = []
            for cur_asgn, cur_cstr in zip(asgns, ocstrs):
                index_eq_0 = []
                if cur_cstr['equal_0']:
                    unravel_index = np.unravel_index(
                        cur_cstr['equal_0'],
                        (len(cur_cstr['group_tracklet']), n_actions + 1))
                    for i, id_group in enumerate(unravel_index[0]):
                        id_class = unravel_index[1][i]
                        ravel_idx = [
                            (n_actions + 1) * x + id_class
                            for x in cur_cstr['group_tracklet'][id_group]
                        ]
                        index_eq_0 += ravel_idx
                cstrs.append({'equal_0': index_eq_0})
        else:
            cstrs = ocstrs
        # Do MAP custom evaluation.

        # concat all test sample scores and their gts
        all_gt_test = np.concatenate(gts)
        all_gt_test[all_gt_test > 0] = 1
        all_scores_test_list = [
            np.dot(
                np.append(x, 100.0 * np.ones([x.shape[0], 1]), axis=1),
                weights) for x in feats
        ]
        all_scores_test = np.concatenate(all_scores_test_list)

        # set test scores forced to bck to -Inf
        if not isVal:
            mask_background_list = []
            for i in range(len(asgns)):
                cur_mask = np.ones(asgns[i].shape)
                if cstrs[i]['equal_0']:
                    cur_mask[np.unravel_index(cstrs[i]['equal_0'],
                                             asgns[i].shape)] = 0.0
                mask_background_list.append(cur_mask)

            all_scores_test_modified_list = [
               np.multiply(all_scores_test_list[i], mask_background_list[i]) -
               100000 * (1 - mask_background_list[i]) for i in range(len(asgns))
            ]
            all_scores_test_mod = np.concatenate(all_scores_test_modified_list)


        # get mAP from different sources
        mAP_string = ''
        if isVal:
            mAP_string += 'VAL: '
        save_dic = {}
        # XW
        average_precision = np.zeros(n_actions)
        list_pr = []
        for i in range(n_actions):
            average_precision[i] = average_precision_score(
                all_gt_test[:, i], all_scores_test[:, i])
            precision, recall, _ = precision_recall_curve(all_gt_test[:, i], all_scores_test[:, i])
            list_pr.append((precision, recall))
        mAP_string += 'MAP XW: {}'.format(average_precision.mean())
        save_dic['average_precision_XW'] = average_precision
        if self.save_pr_curves:
            save_dic['list_pr_XW'] = list_pr

        # XW with cstr: Rescore XW only where Z is non zero.
        if not isVal:
            average_precision_modified = np.zeros(n_actions)
            list_pr_mod = []
            for i in range(n_actions):
                average_precision_modified[i] = average_precision_score(
                    all_gt_test[:, i], all_scores_test_mod[:, i])
                precision, recall, _ = precision_recall_curve(all_gt_test[:, i], all_scores_test_mod[:, i])
                list_pr_mod.append((precision, recall))
            mAP_string += ', MAP XW w. cstrs: {}'.format(average_precision_modified.mean())
            save_dic['average_precision_modXW'] = average_precision_modified
            if self.save_pr_curves:
                save_dic['list_pr_modXW'] = list_pr_mod

        # Z
        if not isVal:
            average_precision_asgn = np.zeros(n_actions)
            asgns_concat = np.concatenate(asgns)
            # Break potential ties
            asgns_concat += 1e-7 * np.random.randn(asgns_concat.shape[0], asgns_concat.shape[1])
            list_pr_asgn = []
            for i in range(n_actions):
                average_precision_asgn[i] = average_precision_score(
                   all_gt_test[:, i], asgns_concat[:, i])
                precision, recall, _ = precision_recall_curve(all_gt_test[:, i], asgns_concat[:, i])
                list_pr_asgn.append((precision, recall))
            mAP_string += ', MAP Z: {}'.format(average_precision_asgn.mean())
            save_dic['average_precision_Z'] = average_precision_asgn
            if self.save_pr_curves:
                save_dic['list_pr_Z'] = list_pr_asgn

        print mAP_string

        if self.path_save is not None:
            patsave = ''
            if isVal:
                patsave = 'VAL_'
            patsave += '{:06d}.pkl'
            with open(os.path.join(self.path_save, patsave.format(self.call_number)), 'wb') as f:
                pickle.dump(save_dic, f)

        return average_precision.mean()
