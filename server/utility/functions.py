from typing import Iterable, Set, TypeVar
import numpy as np
from itertools import groupby


def rolling_mean_variance(series, T, weight=1.0):
    L, N = series.shape
    if L < T:
        return np.empty((0, N)), np.empty((0, N))
    init_ws = np.tile(np.array([[weight ** i for i in range(T - 2, -1, -1)]]).transpose(), [1, N])
    mean = np.zeros((L - T + 1, N))
    var = np.zeros((L - T + 1, N))
    s0 = (np.sum(init_ws[:, 0]) + weight ** (T - 1)) * np.ones((1, N))
    s1 = np.sum(series[:T - 1, :] * init_ws, axis=0, keepdims=True)
    s2 = np.sum(series[:T - 1, :] ** 2 * init_ws, axis=0, keepdims=True)
    s1_prev = np.zeros((1, N))
    s2_prev = np.zeros((1, N))
    for i in range(L - T + 1):
        s1 *= weight
        s2 *= weight
        s1 += series[i + T - 1, :] - s1_prev * weight ** T
        s2 += series[i + T - 1, :] ** 2 - s2_prev * weight ** T
        s1_prev = series[i, :]
        s2_prev = series[i, :] ** 2
        mean[i, :] = s1 / s0
        var[i, :] = (s0 * s2 - s1 * s1) / (s0 * (s0 - 1))
    return mean, var


def rolling_window(series, T):
    shape = series.shape[:-1] + (series.shape[-1] - T + 1, T)
    strides = series.strides + (series.strides[-1],)
    return np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)


def rolling_minmax(series, T, minmax='min'):
    if minmax == 'min':
        np_argfun = np.argmin
        np_fun = np.min
    elif minmax == 'max':
        np_argfun = np.argmax
        np_fun = np.max
    else:
        raise Exception('"minmax" should be either "min" or "max".')

    # non-optimal
    return np_fun(rolling_window(series.transpose(), T), axis=1).transpose()


def flatten_prismatic_tensor(K):
    _, N, M = K.shape
    if N != M:
        raise Exception('Two lowest dimensions must be equal.')
    return np.array(list(map(lambda x: x[np.triu_indices(N, 1)], K)))


def calculate_pairwise_diffs(prices):
    days, N = prices.shape
    diffs = np.zeros((days, N, N))
    for i in range(N):
        for j in range(i + 1, N):
            diffs[:, i, j] = prices[:, i] - prices[:, j]
    return flatten_prismatic_tensor(diffs)


def pairs_from_N(N):
    return int(N * (N - 1) / 2)


def N_from_pairs(P):
    return int((np.sqrt(8 * P + 1) + 1) / 2)


def encode_pair(i, j, N):
    if j < i:
        i, j = j, i
    return N * (i - 1) + (j - 1) - int(i * (i + 1) / 2)


def encode_pairs(ijs, N):
    return np.array([encode_pair(i, j, N) for i, j in ijs])


def decode_pair(c, N, inv=False):
    b1 = N + 0.5
    b2 = (N - 0.5) ** 2
    return _decode_pair(c, N, b1, b2, inv)


def decode_pairs(cs, N, inv=False):
    b1 = N + 0.5
    b2 = (N - 0.5) ** 2
    return np.array(
        np.vectorize(lambda c: _decode_pair(c, N, b1, b2, inv), otypes=(np.int32, np.int32))(cs)).transpose()


def _decode_pair(c, N, b1, b2, inv):
    i = int(b1 - np.sqrt(b2 - 2 * c))
    j = c + int(i * (i + 1) / 2) - N * (i - 1) + 1
    if inv:
        return j, i
    else:
        return i, j


def trade_pair(diffs, t, pair, inv=False):
    P = diffs.shape[1]
    N = N_from_pairs(P)

    if isinstance(pair, tuple):
        i, j = tuple
        c = encode_pair(i, j, N)
    elif np.issubdtype(pair, np.integer):
        c = pair
    else:
        raise Exception('"pair" should be either tuple or int.')

    return (1 - 2 * inv) * (diffs[t, c] - diffs[t + 1, c]) / 2


def trade_pairs(diffs, ts, pairs, inv=False):
    P = diffs.shape[1]
    N = int(np.sqrt(8 * P + 1) / 2 - 1.5)

    if len(pairs.shape) == 2:
        cs = encode_pairs(pairs, N)
    elif len(pairs.shape) == 1:
        cs = pairs
    else:
        raise Exception('"pairs" should be one- or two-dimensional.')

    return np.array([trade_pair(diffs, t, c, inv) for t, c in zip(ts, cs)])


def select_argsort(matrix, argsort):
    n, _ = matrix.shape
    _, l = argsort.shape
    return matrix[np.arange(n).repeat(l), argsort.ravel()].reshape((n, l))


def statistical_arbitrage(diffs, means, varss, p, d,
                          method='thresh-devs', scale=True,
                          return_pairs=False, return_weights=False, return_profits=True):
    T = diffs.shape[0] - means.shape[0] + 1
    P = diffs.shape[1]
    L = means.shape[0]
    N = N_from_pairs(P)

    if 'thresh' not in method:
        selected_codes = np.argsort(varss, axis=1)[:-1, :p]
        selected_diffs = select_argsort(diffs[T - 1:-1, :], selected_codes)
        selected_means = select_argsort(means[:-1, :], selected_codes)
        selected_vars = select_argsort(varss[:-1, :], selected_codes)

    if method == 'thresh-devs':
        if p is not None:
            selected_codes = varss[:-1, :] <= p
        else:
            selected_codes = np.ones_like(varss[:-1, :], np.bool)
        ts, cs = np.nonzero(selected_codes)
        selected_means = means[:-1, :][selected_codes]
        selected_vars = varss[:-1, :][selected_codes]
        lower_bounds = selected_means - d * np.sqrt(selected_vars)
        upper_bounds = selected_means + d * np.sqrt(selected_vars)
        selected_diffs = diffs[T - 1:-1, :][selected_codes]
        del (selected_means, selected_vars)
        invs = np.less(selected_diffs, lower_bounds)
        no_invs = np.greater(selected_diffs, upper_bounds)
        del (selected_diffs, upper_bounds, lower_bounds)
        ts_invs = ts[invs] + T - 1
        ts_no_invs = ts[no_invs] + T - 1
        invs = cs[invs]
        no_invs = cs[no_invs]
        del (ts, cs)

    elif method == 'devs':
        lower_bounds = selected_means - d * np.sqrt(selected_vars)
        upper_bounds = selected_means + d * np.sqrt(selected_vars)
        ts_invs, invs = np.nonzero(np.less(selected_diffs, lower_bounds))
        ts_no_invs, no_invs = np.nonzero(np.greater(selected_diffs, upper_bounds))
        invs = selected_codes[ts_invs, invs]
        no_invs = selected_codes[ts_no_invs, no_invs]
        ts_invs += T - 1
        ts_no_invs += T - 1

    elif method == 'max-abs-devs':
        max_abs_devs = select_argsort(
            np.max(np.abs(rolling_window(diffs.T, T).T - means), axis=0)[:-1, :],
            selected_codes)
        lower_bounds = selected_means - d * max_abs_devs
        upper_bounds = selected_means + d * max_abs_devs
        ts_invs, invs = np.nonzero(np.less(selected_diffs[1:, :], lower_bounds[:-1, :]))
        ts_no_invs, no_invs = np.nonzero(np.greater(selected_diffs[1:, :], upper_bounds[:-1, :]))
        ts_invs += 1
        ts_no_invs += 1
        invs = selected_codes[ts_invs, invs]
        no_invs = selected_codes[ts_no_invs, no_invs]
        ts_invs += T - 1
        ts_no_invs += T - 1

    else:
        raise Exception('Method not supported.')

    invs_profit = trade_pairs(diffs, ts_invs, invs, True)
    no_invs_profit = trade_pairs(diffs, ts_no_invs, no_invs, False)
    ts_total = np.append(ts_invs, ts_no_invs)
    ts_order = np.argsort(ts_total)
    profit = np.append(invs_profit, no_invs_profit)[ts_order]
    ts_total = ts_total[ts_order]

    if return_pairs:
        pairs = np.vstack((decode_pairs(invs, N, inv=True),
                           decode_pairs(no_invs, N, inv=False)))[ts_order, ...]
        if return_weights:
            weights = np.abs(np.hstack(((diffs[ts_invs, invs] - means[ts_invs - T + 1, invs])
                                        / np.sqrt(varss[ts_invs - T + 1, invs]),
                                        (diffs[ts_no_invs, no_invs] - means[ts_no_invs - T + 1, no_invs])
                                        / np.sqrt(varss[ts_no_invs - T + 1, no_invs])))[ts_order, ...]) - d

    if scale:
        scales = [len(list(it)) for _, it in groupby(ts_total)]
        last_ts = None
        j = -1
        for i in range(len(profit)):
            if ts_total[i] != last_ts:
                last_ts = ts_total[i]
                j += 1
            profit[i] /= scales[j]

    return_values = (ts_total,)
    if return_profits:
        return_values += (profit,)
    if return_pairs:
        return_values += (pairs,)
        if return_weights:
            return_values += (weights,)
    return return_values


def partition_as(days, pairs, weights=None):
    lens = [len(list(it)) for _, it in groupby(days)]
    i = 0
    j = 0
    n = len(pairs)
    while i != n:
        if weights is not None:
            yield days[i], pairs[i:i + lens[j], ...], weights[i:i + lens[j], ...]
        else:
            yield days[i], pairs[i:i + lens[j], ...]
        i += lens[j]
        j += 1


def calculate_preference_flow(pairs, weights=None, scale=False, fast=True, return_inconsistency=False):
    nodes = set(np.ravel(pairs))
    M = len(nodes)
    P = pairs_from_N(M)
    nodes = sorted(list(nodes))
    node_index = {node: index for index, node in enumerate(nodes)}
    A = np.zeros((P, M))
    F = np.zeros((P, 1))
    # for ii in range(M):
    #     for ji in range(ii + 1, M):
    #         c = ii * M + ji - int((ii + 1) * (ii + 2) / 2)
    #         A[c, ii] = 1
    #         A[c, ji] = -1
    if weights is None:
        for i, j in pairs:
            ii = node_index[i]
            ji = node_index[j]
            m = min(ii, ji)
            n = max(ii, ji)
            c = m * M + n - int((m + 1) * (m + 2) / 2)
            A[c, ii] = 1
            A[c, ji] = -1
            F[c] = 1
    else:
        for (i, j), weight in zip(pairs, weights):
            ii = node_index[i]
            ji = node_index[j]
            m = min(ii, ji)
            n = max(ii, ji)
            c = m * M + n - int((m + 1) * (m + 2) / 2)
            A[c, ii] = 1
            A[c, ji] = -1
            F[c] = weight

    if fast:
        valids = np.where(F != 0)[0]
        F = F[valids]
        A = A[valids]
        X = np.dot(A.T, F) / M
    else:
        X = np.dot(np.linalg.inv(np.dot(A.T, A) + np.ones((M, M))), np.dot(A.T, F))

    if scale:
        X *= M

    X_dict = {node: X[index, 0] for index, node in enumerate(nodes)}

    if return_inconsistency:
        inconsistency = np.linalg.norm(np.dot(A, X)) / np.linalg.norm(F)
        return X_dict, inconsistency
    else:
        return X_dict


def trade_single(log_prices, t, i, inv=False):
    """inv: False ako shortamo, True ako kupujemo."""
    return (1 - 2 * inv) * (log_prices[t, i - 1] - log_prices[t + 1, i - 1])


def trade_singles(log_prices, ts, iss, inv=False):
    return (1 - 2 * inv) * (log_prices[ts, iss - 1] - log_prices[ts + 1, iss - 1])


T = TypeVar('T')


def turnover_ratio(series: Iterable[Set[T]]):
    turnovers = []
    S1 = set()
    for S2 in series:
        lS1 = len(S1)
        lS2 = len(S2)
        turnover = 0
        if lS1:
            turnover += len(S1 - S2) / lS1
        if lS2:
            turnover += len(S2 - S1) / lS2
        if lS1 and lS2:
            turnover += len(S1 & S2) * abs(1 / lS1 - 1 / lS2)
        turnovers += [turnover]
        S1 = S2
    return np.array(turnovers)


def join_ts_profit(ts, profits):
    if not len(ts):
        return [], []
    pprofits = partition_as(ts, profits)
    return np.array([[t, np.sum(profit, axis=0)] for t, profit in pprofits]).transpose()
