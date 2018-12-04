""" Implements BCFW for DIFFRAC objectives. """

import numpy as np
import os
from tqdm import tqdm
from numpy.linalg import norm as matrix_norm
import time


def get_feat_block(feats, block_idx, memory_mode, bias_value=-1.0):
    """Get feature for a given block."""
    if memory_mode == 'RAM':
        feat = feats[block_idx]
    elif memory_mode == 'disk':
        feat = np.load(feats[block_idx])
    else:
        raise ValueError(
            'Memory mode {} is not supported.'.format(memory_mode))

    if bias_value > 0.0:
        feat = np.append(
            feat, bias_value * np.ones([feat.shape[0], 1]), axis=1)

    return feat


def get_p_block(p_matrix, block_idx, memory_mode):
    if memory_mode == 'RAM':
        return p_matrix[block_idx]
    elif memory_mode == 'disk':
        return np.load(p_matrix[block_idx])
    else:
        raise ValueError(
            'Memory mode {} is not supported.'.format(memory_mode))


def compute_p_matrix(feats, alpha, memory_mode, bias_value=-1.0):
    """Precompute the P dictionnary matrix."""
    _, d = np.shape(
        get_feat_block(feats, 0, memory_mode, bias_value=bias_value))

    # Compute X^TX
    print('Computing xtx...')
    x_t_x = np.zeros([d, d])
    N = 0
    for i in tqdm(range(len(feats))):
        x = get_feat_block(feats, i, memory_mode, bias_value=bias_value)
        x_t_x += np.dot(np.transpose(x), x)
        N += x.shape[0]

    # Compute P
    p_matrix = []
    print('Inverting big matrix...')
    inv_mat = np.linalg.inv(x_t_x + N * alpha * np.eye(d))
    print('Computing P matrix by block...')
    for i in tqdm(range(len(feats))):
        x = get_feat_block(feats, i, memory_mode, bias_value=bias_value)
        sol = np.dot(inv_mat, np.transpose(x))
        if memory_mode == 'RAM':
            p_matrix.append(np.array(sol))
        else:
            path_x = feats[i]
            base_path, filename = os.path.split(path_x)
            np.save(os.path.join(base_path, 'P_{}'.format(filename)), sol)
            p_matrix.append(path_x)

    return p_matrix, N


def compute_weights(p_matrix, asgn, memory_mode):
    d, _ = np.shape(get_p_block(p_matrix, 0, memory_mode))
    _, k = np.shape(asgn[0])

    weights = np.zeros([d, k])

    print('Computing weights from scratch...')
    for i in tqdm(range(len(p_matrix))):
        weights += np.dot(get_p_block(p_matrix, i, memory_mode), asgn[i])

    return weights


def compute_obj(x, y, weights, n_feats):
    return 1.0 / n_feats * matrix_norm(np.dot(x, weights) - y, ord='fro')**2


def compute_grad(x, y, weights, n_feats):
    return 1.0 / n_feats * (y - np.dot(x, weights))


def compute_gap(x,
                y,
                weights,
                n_feats,
                cstr,
                cstr_solver,
                opt_y=None,
                grad_y=None):

    # Check if we need to call the oracle.
    if opt_y is None:
        grad_y = compute_grad(x, y, weights, n_feats)
        opt_y = cstr_solver.solve(cstr, grad_y)

    gap = -np.multiply(opt_y - y, grad_y).sum()

    return gap


def sample_block(gaps, block_sampling):
    if block_sampling == 'uniform':
        return np.random.randint(0, len(gaps), 1)[0]
    elif block_sampling == 'gap_sampling':
        if not np.all(gaps >= 0):
            print('Warning: some gaps are negative block {}, value :{}'.format(
                gaps.argmin(), gaps.min()))
            gaps[gaps < 0] = 0.00000001

        gap_prob = gaps / gaps.sum()
        return np.random.choice(len(gaps), 1, p=gap_prob)[0]


def display_information(iter,
                        max_iter,
                        gaps,
                        eval_metric,
                        objective_value=None,
                        verbose='silent',
                        prev_time=-1,
                        prev_global_time=-1):
    """Display information about the training."""
    if objective_value is None:
        objective_value = []

    if verbose in ['normal', 'heavy']:
        string_display = 'Iteration {0:05d}/{1:05d}, Gap sum: {2:.4E}'.format(
            iter, max_iter, gaps.sum())

        new_time = time.time()
        if prev_time > 0:
            diff_time = int(round(new_time - prev_time))
            string_display += ' ({:d} s)'.format(diff_time)
        if prev_global_time > 0:
            diff_time = int(round(new_time - prev_global_time))
            string_display += ' (Glob. {:d} s)'.format(diff_time)

        if eval_metric >= 0:
            string_display += ', Eval metric: {:.2f}'.format(eval_metric)

        if objective_value:
            string_display += ', Objective: '
            string_display += ','.join([
                '{}: {:.4E}'.format(key, value)
                for key, value in objective_value.items()
            ])

        print(string_display)


def save_asgn_block(path_save_asgn, block_idx, asgn, t):
    np.save(
        os.path.join(path_save_asgn, '{0}_{1:05d}.npy'.format(block_idx, t)),
        asgn[block_idx])


def save_xw_block(path_save_asgn, block_idx, x, weights, t):
    np.save(
        os.path.join(path_save_asgn, 'xw_{0}_{1:05d}.npy'.format(block_idx,
                                                                 t)),
        np.dot(x, weights))


def save_gt_block(path_save_asgn, block_idx, gts):
    np.save(
        os.path.join(path_save_asgn, '{}_gt.npy'.format(block_idx)),
        gts[block_idx])


def solver(feats,
           asgn,
           cstrs,
           cstrs_solver,
           gts=None,
           eval_function=None,
           rounding_function=None,
           alpha=1e-4,
           memory_mode='RAM',
           bias_value=-1.0,
           n_iterations=10000,
           block_sampling='uniform',
           verbose='silent',
           gap_frequency=2000,
           eval_frequency=500,
           verbose_frequency=250,
           objective_frequency=250,
           path_save_asgn=None,
           validation_info=None):
    """Main solver for DiffracBCFW.

    Args:
        feats: Input features as a list (one entry per block).
        asgn: Assignment variables as a list (one entry per block). This provides
            the initialization of the system.
        cstrs: Input constraints as a dictionary (one entry per block).
        cstrs_solver: Method that takes as input a gradient for a block and a cstrs and then
            returns the LP solution.
        gts: A ground truth can be specified if you wish to evaluate your solution.
        eval_function: an eval function method can be provided.
        rounding_function: rounding function.
        alpha: Value of the regularization parameter (lambda in the paper).
        memory_mode: `disk` (features are stored in disk) or `RAM` (features are in RAM).
        bias_value: Value to add for the bias (if negative no bias is added to the features).
        n_iterations: Number of iterations of the solver.
        block_sampling: Method for sampling block.
        verbose: `silent`, `normal`, `heavy`.
        gap_frequency: frequency to recompute all the gaps.
        eval_frequency: frequency to perform evaluation.
        verbose_frequency: frequency to print info.
        objective_frequency: frequency to compute objective (only used if positive).
        path_save_asgn: If not None save asgn at path_save_asgn. None by default.
        validation_info: If not None perform evaluation on validation
    """

    compute_objective = False
    objective_value = None
    if objective_frequency > 0:
        compute_objective = True

    save_asgn = False
    save_ids = []
    if path_save_asgn is not None:
        if not os.path.exists(path_save_asgn):
            os.makedirs(path_save_asgn)
        # Monitor evolution of asgn during optim on a subset of samples.
        save_asgn = True
        n_save_asgn = min(20, len(asgn))
        save_ids = np.random.choice(len(asgn), n_save_asgn, replace=False)

    # Pre-compute the P matrix.
    p_matrix, n_feats = compute_p_matrix(
        feats, alpha, memory_mode, bias_value=bias_value)

    # Compute W.
    weights = compute_weights(p_matrix, asgn, memory_mode=memory_mode)

    # Init the gaps.
    gaps = np.zeros(len(feats))
    print('Computing init gaps...')
    for block_idx in tqdm(range(len(feats))):
        x = get_feat_block(
            feats, block_idx, memory_mode, bias_value=bias_value)
        gaps[block_idx] = compute_gap(x, asgn[block_idx], weights, n_feats,
                                      cstrs[block_idx], cstrs_solver)

        if save_asgn and block_idx in save_ids:
            save_asgn_block(path_save_asgn, block_idx, asgn, 0)
            save_xw_block(path_save_asgn, block_idx, x, weights, 0)
            save_gt_block(path_save_asgn, block_idx, gts)

    print('Init gap: {0:4E}, starting the optimization...'.format(gaps.sum()))

    eval_metric = -1.0
    prev_time = time.time()  # init time of iterations
    prev_global_time = prev_time
    for t in range(n_iterations):
        if eval_frequency > 0 and t % eval_frequency == 0:
            # Evaluation.
            if eval_function is not None and gts is not None:
                print('Performing evaluation...')
                eval_metric = eval_function.evaluate(asgn, gts, weights, feats,
                                                     rounding_function, cstrs)
            if validation_info is not None:
                gts_val = validation_info['gts']
                feats_val = validation_info['feats']
                eval_function.evaluate(None, gts_val, weights, feats_val,
                                       rounding_function, None)
        else:
            eval_metric = -1.0

        if compute_objective and t % objective_frequency == 0:
            print('Computing objective...')
            objective_value = {}
            # Compute the diffrac objective.
            dfrac_obj = 0.0
            # Data dependent term: 1.0 / N * ||X * W - Y||_2^2
            for block_idx in range(len(feats)):
                x = get_feat_block(
                    feats, block_idx, memory_mode, bias_value=bias_value)
                dfrac_obj += compute_obj(x, asgn[block_idx], weights, n_feats)

            # Regularization term: \alpha * || W ||_2^2
            dfrac_obj += alpha * matrix_norm(weights, ord='fro')**2
            objective_value['dfrac'] = dfrac_obj

        # Print information.
        if t % verbose_frequency == 0:
            display_information(t, n_iterations, gaps, eval_metric,
                                objective_value, verbose, prev_time, prev_global_time)
            prev_time = time.time()

        # Sample a block.
        block_idx = sample_block(gaps, block_sampling)
        # Compute gradient.
        x = get_feat_block(
            feats, block_idx, memory_mode, bias_value=bias_value)
        y = asgn[block_idx]

        grad_y = compute_grad(x, y, weights, n_feats)

        opt_y = cstrs_solver.solve(cstrs[block_idx], grad_y)
        gaps[block_idx] = compute_gap(x, y, weights, n_feats,
                                      cstrs[block_idx], cstrs_solver,
                                      opt_y, grad_y)

        # Step size computation.
        p = get_p_block(p_matrix, block_idx, memory_mode)
        dir_y = opt_y - y
        gamma_n = gaps[block_idx]

        gamma_d = 1.0 / n_feats * np.multiply(
            dir_y, dir_y - np.linalg.multi_dot([x, p, dir_y])).sum()

        gamma = min(1.0, gamma_n / gamma_d)
        # gamma should always be positive.
        if gamma < 0:
            print 'Warning: gamma = {}, gap_i = {}'.format(
                   gamma, gaps[block_idx])
            gamma = 0.0

        # Update variables.
        asgn[block_idx] += gamma * dir_y
        weights += gamma * np.dot(p, dir_y)

        if save_asgn and block_idx in save_ids:
            save_asgn_block(path_save_asgn, block_idx, asgn, t)
            save_xw_block(path_save_asgn, block_idx, x, weights, t)

        # Update gaps if needed.
        if (t + 1) % gap_frequency == 0:
            print('Recomputing gaps...')
            for block_idx in tqdm(range(len(feats))):
                x = get_feat_block(
                    feats, block_idx, memory_mode, bias_value=bias_value)
                gaps[block_idx] = compute_gap(x, asgn[block_idx], weights,
                                              n_feats, cstrs[block_idx],
                                              cstrs_solver)
            display_information(t, n_iterations, gaps, eval_metric,
                                objective_value, verbose)

    return asgn, weights
