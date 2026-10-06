"""Unified registered numerical operators."""
from ._definition import Operator, OPERATOR_REGISTRY, operator
from bagelquant_core.operator.basic import diff, prediction_signal, identity, negate, pct_change, pct_change_frame
from bagelquant_core.operator.boxcox import boxcox
from bagelquant_core.operator.combination import mask, project
from bagelquant_core.operator.comparison import not_
from bagelquant_core.operator.regression import rolling_elastic_net, rolling_lasso, rolling_ols, rolling_ridge
from bagelquant_core.operator.scaling import vol_scale
from bagelquant_core.operator.cross_sectional import group_demean, group_max, group_mean, group_median, group_min, group_percentile, group_rank, group_rankpct, group_std, group_zscore, orthogonalize
from bagelquant_core.operator.temporal import canonicalize_values, project_domain, constant, date_age_constraint, denoise, lag, negonly, notnan, posonly, remove_repeated, replace_inf
from bagelquant_core.operator.kelly import kelly, kelly_nonan_standardize, kelly_rank_boxcox, kelly_rescaling_weight
from bagelquant_core.operator.logarithmic import inv_log_sqrt_rank, log, log1p, log_rank, signed_log1p
from bagelquant_core.operator.missing import bfill, ffill, fillna, fillna_zero
from bagelquant_core.operator.normalization import min_max_scale, net_scale, normalize, rank, winsorize, zscore
from bagelquant_core.operator.outlier import trim, trim_quantile, truncate
from bagelquant_core.operator.power import power, signed_power, sqrt
from bagelquant_core.operator.ranking import nrank, rankpct
from bagelquant_core.operator.replace import non_nan_to_one, non_nan_to_zero, replace_non_nan
from bagelquant_core.operator.rolling_stats import ewm_mean, ewm_std, ewm_var, rolling_ewm_fw, rolling_kurt, rolling_max, rolling_mean, rolling_median, rolling_min, rolling_percentile, rolling_rank, rolling_skew, rolling_std, rolling_sum, rolling_var, rolling_zscore, smooth
from bagelquant_core.operator.sign import abs, ceil, sign
from bagelquant_core.operator.streaks import diff_from_last_change, pct_change_from_last_change, repeat_count, streak_count
from bagelquant_core.operator.variance_stabilization import anscombe, freeman
from bagelquant_core.operator.translation import demean, translate_to_pos
from bagelquant_core.operator.trigonometric import arccos, arcsin, arctan, arctanh, cos, sin, trig
from bagelquant_core.operator.aggregation import maximum, mean, minimum, product, sum_frames, weighted_mean, weighted_sum
from bagelquant_core.operator.arithmetic import add, div, mul, sub
from bagelquant_core.operator.combination import broadcast_by_time, coalesce
from bagelquant_core.operator.comparison import and_, equal, greater, greater_equal, less, less_equal, or_, power_df, xand, xor
from bagelquant_core.operator.regression import rolling_corr, rolling_cov

from .portfolio import top_n, equal_weight, rebalance, regularized_weights, exposure_constrained_weights
from .training import rolling_elastic_net_prediction, rolling_lightgbm_prediction
from . import prediction as _prediction_models  # noqa: F401 — registers model operators
from bagelquant_core._documentation import operation_category, operation_description

__all__ = ['OPERATOR_REGISTRY', 'Operator', 'prediction_signal', 'operation_category', 'operation_description', 'abs', 'add', 'and_', 'anscombe', 'arccos', 'arcsin', 'arctan', 'arctanh', 'bfill', 'boxcox', 'broadcast_by_time', 'canonicalize_values', 'ceil', 'coalesce', 'constant', 'cos', 'date_age_constraint', 'demean', 'denoise', 'diff', 'diff_from_last_change', 'div', 'equal', 'ewm_mean', 'ewm_std', 'ewm_var', 'ffill', 'fillna', 'fillna_zero', 'freeman', 'greater', 'greater_equal', 'group_demean', 'group_max', 'group_mean', 'group_median', 'group_min', 'group_percentile', 'group_rank', 'group_rankpct', 'group_std', 'group_zscore', 'identity', 'inv_log_sqrt_rank', 'kelly', 'kelly_nonan_standardize', 'kelly_rank_boxcox', 'kelly_rescaling_weight', 'lag', 'less', 'less_equal', 'log', 'log1p', 'log_rank', 'mask', 'maximum', 'mean', 'min_max_scale', 'minimum', 'mul', 'negate', 'negonly', 'net_scale', 'non_nan_to_one', 'non_nan_to_zero', 'normalize', 'not_', 'notnan', 'nrank', 'operator', 'or_', 'orthogonalize', 'pct_change', 'pct_change_frame', 'pct_change_from_last_change', 'posonly', 'power', 'power_df', 'product', 'project', 'project_domain', 'rank', 'rankpct', 'remove_repeated', 'repeat_count', 'replace_inf', 'replace_non_nan', 'rolling_corr', 'rolling_cov', 'rolling_elastic_net', 'rolling_ewm_fw', 'rolling_kurt', 'rolling_lasso', 'rolling_max', 'rolling_mean', 'rolling_median', 'rolling_min', 'rolling_ols', 'rolling_percentile', 'rolling_rank', 'rolling_ridge', 'rolling_skew', 'rolling_std', 'rolling_sum', 'rolling_var', 'rolling_zscore', 'sign', 'signed_log1p', 'signed_power', 'sin', 'smooth', 'sqrt', 'streak_count', 'sub', 'sum_frames', 'translate_to_pos', 'trig', 'trim', 'trim_quantile', 'truncate', 'vol_scale', 'weighted_mean', 'weighted_sum', 'winsorize', 'xand', 'xor', 'zscore']

__all__ += ["top_n", "equal_weight", "rebalance", "regularized_weights", "exposure_constrained_weights",
            "rolling_elastic_net_prediction", "rolling_lightgbm_prediction"]
for _kind in ("identity", "equal_weight", "ic_weighted", "ic_weighted_decay", "quantile_ic_weighted", "ols", "gls"):
    _name = _kind + "_prediction"
    globals()[_name] = OPERATOR_REGISTRY.get("prediction:" + _kind)
    __all__.append(_name)

