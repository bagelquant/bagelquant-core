# Transformer reference

Each public transformer has a generated reference page with an exact signature,
parameter contract, executable Panel example, and temporal semantics.

## Element-wise

- [`abs`](./abs.md)
- [`anscombe`](./anscombe.md)
- [`arccos`](./arccos.md)
- [`arcsin`](./arcsin.md)
- [`arctan`](./arctan.md)
- [`arctanh`](./arctanh.md)
- [`boxcox`](./boxcox.md)
- [`ceil`](./ceil.md)
- [`constant`](./constant.md)
- [`cos`](./cos.md)
- [`date_age_constraint`](./date_age_constraint.md)
- [`denoise`](./denoise.md)
- [`freeman`](./freeman.md)
- [`identity`](./identity.md)
- [`inv_log_sqrt_rank`](./inv_log_sqrt_rank.md)
- [`kelly`](./kelly.md)
- [`kelly_nonan_standardize`](./kelly_nonan_standardize.md)
- [`kelly_rank_boxcox`](./kelly_rank_boxcox.md)
- [`kelly_rescaling_weight`](./kelly_rescaling_weight.md)
- [`log`](./log.md)
- [`log1p`](./log1p.md)
- [`log_rank`](./log_rank.md)
- [`negate`](./negate.md)
- [`negonly`](./negonly.md)
- [`non_nan_to_one`](./non_nan_to_one.md)
- [`non_nan_to_zero`](./non_nan_to_zero.md)
- [`notnan`](./notnan.md)
- [`nrank`](./nrank.md)
- [`posonly`](./posonly.md)
- [`power`](./power.md)
- [`replace_non_nan`](./replace_non_nan.md)
- [`sign`](./sign.md)
- [`signed_log1p`](./signed_log1p.md)
- [`signed_power`](./signed_power.md)
- [`sin`](./sin.md)
- [`sqrt`](./sqrt.md)
- [`translate_to_pos`](./translate_to_pos.md)
- [`trig`](./trig.md)
- [`trim`](./trim.md)
- [`trim_quantile`](./trim_quantile.md)
- [`truncate`](./truncate.md)

## Missing data

- [`bfill`](./bfill.md)
- [`ffill`](./ffill.md)
- [`fillna`](./fillna.md)
- [`fillna_zero`](./fillna_zero.md)
- [`replace_inf`](./replace_inf.md)

## Cross-sectional

- [`demean`](./demean.md)
- [`min_max_scale`](./min_max_scale.md)
- [`net_scale`](./net_scale.md)
- [`normalize`](./normalize.md)
- [`rank`](./rank.md)
- [`rankpct`](./rankpct.md)
- [`winsorize`](./winsorize.md)
- [`zscore`](./zscore.md)

## Rolling statistics

- [`diff`](./diff.md)
- [`diff_from_last_change`](./diff_from_last_change.md)
- [`ewm_mean`](./ewm_mean.md)
- [`ewm_std`](./ewm_std.md)
- [`ewm_var`](./ewm_var.md)
- [`lag`](./lag.md)
- [`pct_change`](./pct_change.md)
- [`pct_change_from_last_change`](./pct_change_from_last_change.md)
- [`remove_repeated`](./remove_repeated.md)
- [`repeat_count`](./repeat_count.md)
- [`rolling_ewm_fw`](./rolling_ewm_fw.md)
- [`rolling_kurt`](./rolling_kurt.md)
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
- [`streak_count`](./streak_count.md)

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

## Masking & scaling

- [`mask`](./mask.md)
- [`project`](./project.md)
- [`vol_scale`](./vol_scale.md)

## Logical & comparison

- [`not_`](./not_.md)

## Rolling regression

- [`rolling_elastic_net`](./rolling_elastic_net.md)
- [`rolling_lasso`](./rolling_lasso.md)
- [`rolling_ols`](./rolling_ols.md)
- [`rolling_ridge`](./rolling_ridge.md)
