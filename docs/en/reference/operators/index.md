# Operator reference

All calls accept typed Nodes and return a deferred Node. Scalar parameters never become peer inputs.

## Aggregation

- [`maximum`](./maximum.md)
- [`mean`](./mean.md)
- [`minimum`](./minimum.md)
- [`product`](./product.md)
- [`sum_frames`](./sum_frames.md)
- [`weighted_mean`](./weighted_mean.md)
- [`weighted_sum`](./weighted_sum.md)

## Arithmetic

- [`add`](./add.md)
- [`div`](./div.md)
- [`mul`](./mul.md)
- [`power`](./power.md)
- [`power_df`](./power_df.md)
- [`sub`](./sub.md)

## Cross-sectional

- [`anscombe`](./anscombe.md)
- [`demean`](./demean.md)
- [`inv_log_sqrt_rank`](./inv_log_sqrt_rank.md)
- [`log_rank`](./log_rank.md)
- [`min_max_scale`](./min_max_scale.md)
- [`net_scale`](./net_scale.md)
- [`normalize`](./normalize.md)
- [`nrank`](./nrank.md)
- [`rank`](./rank.md)
- [`rankpct`](./rankpct.md)
- [`translate_to_pos`](./translate_to_pos.md)
- [`trim_quantile`](./trim_quantile.md)
- [`winsorize`](./winsorize.md)
- [`zscore`](./zscore.md)

## Element-wise

- [`abs`](./abs.md)
- [`arccos`](./arccos.md)
- [`arcsin`](./arcsin.md)
- [`arctan`](./arctan.md)
- [`arctanh`](./arctanh.md)
- [`boxcox`](./boxcox.md)
- [`broadcast_by_time`](./broadcast_by_time.md)
- [`canonicalize_values`](./canonicalize_values.md)
- [`ceil`](./ceil.md)
- [`constant`](./constant.md)
- [`cos`](./cos.md)
- [`date_age_constraint`](./date_age_constraint.md)
- [`denoise`](./denoise.md)
- [`equal_weight`](./equal_weight.md)
- [`exposure_constrained_weights`](./exposure_constrained_weights.md)
- [`freeman`](./freeman.md)
- [`identity`](./identity.md)
- [`log`](./log.md)
- [`log1p`](./log1p.md)
- [`negate`](./negate.md)
- [`negonly`](./negonly.md)
- [`non_nan_to_one`](./non_nan_to_one.md)
- [`non_nan_to_zero`](./non_nan_to_zero.md)
- [`notnan`](./notnan.md)
- [`posonly`](./posonly.md)
- [`prediction_signal`](./prediction_signal.md)
- [`rebalance`](./rebalance.md)
- [`regularized_weights`](./regularized_weights.md)
- [`replace_non_nan`](./replace_non_nan.md)
- [`sign`](./sign.md)
- [`signed_log1p`](./signed_log1p.md)
- [`signed_power`](./signed_power.md)
- [`sin`](./sin.md)
- [`sqrt`](./sqrt.md)
- [`top_n`](./top_n.md)
- [`trig`](./trig.md)
- [`trim`](./trim.md)
- [`truncate`](./truncate.md)

## Group & neutralization

- [`group_demean`](./group_demean.md)
- [`group_max`](./group_max.md)
- [`group_mean`](./group_mean.md)
- [`group_median`](./group_median.md)
- [`group_min`](./group_min.md)
- [`group_percentile`](./group_percentile.md)
- [`group_rank`](./group_rank.md)
- [`group_rankpct`](./group_rankpct.md)
- [`group_std`](./group_std.md)
- [`group_zscore`](./group_zscore.md)
- [`orthogonalize`](./orthogonalize.md)

## Logical & comparison

- [`and_`](./and_.md)
- [`equal`](./equal.md)
- [`greater`](./greater.md)
- [`greater_equal`](./greater_equal.md)
- [`less`](./less.md)
- [`less_equal`](./less_equal.md)
- [`not_`](./not_.md)
- [`or_`](./or_.md)
- [`xand`](./xand.md)
- [`xor`](./xor.md)

## Masking & scaling

- [`mask`](./mask.md)
- [`project`](./project.md)
- [`project_domain`](./project_domain.md)
- [`vol_scale`](./vol_scale.md)

## Missing data

- [`bfill`](./bfill.md)
- [`coalesce`](./coalesce.md)
- [`ffill`](./ffill.md)
- [`fillna`](./fillna.md)
- [`fillna_zero`](./fillna_zero.md)
- [`replace_inf`](./replace_inf.md)

## Prediction models

- [`equal_weight_prediction`](./equal_weight_prediction.md)
- [`gls_prediction`](./gls_prediction.md)
- [`ic_weighted_decay_prediction`](./ic_weighted_decay_prediction.md)
- [`ic_weighted_prediction`](./ic_weighted_prediction.md)
- [`identity_prediction`](./identity_prediction.md)
- [`ols_prediction`](./ols_prediction.md)
- [`quantile_ic_weighted_prediction`](./quantile_ic_weighted_prediction.md)

## Rolling regression

- [`rolling_elastic_net`](./rolling_elastic_net.md)
- [`rolling_lasso`](./rolling_lasso.md)
- [`rolling_ols`](./rolling_ols.md)
- [`rolling_ridge`](./rolling_ridge.md)

## Rolling statistics

- [`diff`](./diff.md)
- [`diff_from_last_change`](./diff_from_last_change.md)
- [`ewm_mean`](./ewm_mean.md)
- [`ewm_std`](./ewm_std.md)
- [`ewm_var`](./ewm_var.md)
- [`kelly`](./kelly.md)
- [`kelly_nonan_standardize`](./kelly_nonan_standardize.md)
- [`kelly_rank_boxcox`](./kelly_rank_boxcox.md)
- [`kelly_rescaling_weight`](./kelly_rescaling_weight.md)
- [`lag`](./lag.md)
- [`pct_change`](./pct_change.md)
- [`pct_change_from_last_change`](./pct_change_from_last_change.md)
- [`remove_repeated`](./remove_repeated.md)
- [`repeat_count`](./repeat_count.md)
- [`rolling_corr`](./rolling_corr.md)
- [`rolling_cov`](./rolling_cov.md)
- [`rolling_elastic_net_prediction`](./rolling_elastic_net_prediction.md)
- [`rolling_ewm_fw`](./rolling_ewm_fw.md)
- [`rolling_kurt`](./rolling_kurt.md)
- [`rolling_lightgbm_prediction`](./rolling_lightgbm_prediction.md)
- [`rolling_max`](./rolling_max.md)
- [`rolling_mean`](./rolling_mean.md)
- [`rolling_median`](./rolling_median.md)
- [`rolling_min`](./rolling_min.md)
- [`rolling_percentile`](./rolling_percentile.md)
- [`rolling_rank`](./rolling_rank.md)
- [`rolling_skew`](./rolling_skew.md)
- [`rolling_std`](./rolling_std.md)
- [`rolling_sum`](./rolling_sum.md)
- [`rolling_var`](./rolling_var.md)
- [`rolling_zscore`](./rolling_zscore.md)
- [`smooth`](./smooth.md)
- [`streak_count`](./streak_count.md)
