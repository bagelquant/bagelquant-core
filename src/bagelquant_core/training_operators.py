"""Causal rolling numerical estimators with explicit label inputs and audits."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
import hashlib
import json
import numpy as np
import polars as pl
from .operator import Operator, OPERATOR_REGISTRY
from .operation_contract import OperationContract, ExecutionMode, InputDensity, TraceRule
from .machine_learning import ZeroPreservingRmsScaler, ElasticNetModel, fit_elastic_net
from .operator_state import restored_operator_state, save_operator_state, operator_execution_calendar
from .estimator_adapters import LightGBMCandidate, LightGBMEstimatorAdapter

_audits: ContextVar[list | None] = ContextVar("numerical_training_audits", default=None)
SAMPLING_VERSION = "date_balanced_keys.v1"


def _stable_priorities(keys, seed):
    """Fixed BLAKE2 asset keys plus vectorized SplitMix64 temporal mixing."""
    seed_bytes = seed.to_bytes(4, "little")
    assets = keys["asset_id"].unique().sort().to_list() if "asset_id" in keys.columns else []
    asset_hashes = {asset:int.from_bytes(hashlib.blake2b(seed_bytes+asset.encode(), digest_size=8).digest(), "little") for asset in assets}
    temporal = keys["time"].cast(pl.Int32).to_numpy().astype(np.uint64)
    state = temporal ^ np.uint64(seed)
    if assets:
        state ^= keys["asset_id"].replace_strict(asset_hashes, return_dtype=pl.UInt64).to_numpy()
    with np.errstate(over="ignore"):
        state = state + np.uint64(0x9E3779B97F4A7C15)
        state = (state ^ (state >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        state = (state ^ (state >> 27)) * np.uint64(0x94D049BB133111EB)
        return state ^ (state >> 31)


def date_balanced_training_keys(keys: pl.DataFrame, *, budget: int, seed: int = 1729) -> pl.DataFrame:
    """Choose unique eligible keys with max-min fair daily quotas and stable ranks.

    The narrow key table is sampled before gathering any feature matrices.
    Hash ordering is fixed by BLAKE2/SplitMix64; row order,
    partitions, batch size and native thread count have no role in selection.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    keys = keys.select("time", "asset_id").sort("time", "asset_id")
    if keys.select(pl.struct("time", "asset_id").is_duplicated().any()).item():
        raise ValueError("training keys must be unique")
    if keys.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("training keys must be complete")
    if keys.height <= budget:
        return keys
    daily = keys.group_by("time").len().sort("time")
    counts = daily["len"].to_numpy()
    lower, upper = 0, int(counts.max())
    while lower < upper:
        mid = (lower + upper + 1) // 2
        if int(np.minimum(counts, mid).sum()) <= budget:
            lower = mid
        else:
            upper = mid - 1
    quotas = np.minimum(counts, lower)
    remaining = budget - int(quotas.sum())
    daily = daily.with_columns(pl.Series("quota", quotas), pl.Series("priority", _stable_priorities(daily, seed)))
    extra = daily.filter(pl.col("quota") < pl.col("len")).sort("priority", "time").head(remaining)["time"]
    daily = daily.with_columns((pl.col("quota") + pl.col("time").is_in(extra.implode()).cast(pl.UInt32)).alias("quota"))
    return (
        keys.with_columns(pl.Series("priority", _stable_priorities(keys, seed)))
        .sort("time", "priority", "asset_id")
        .with_columns(pl.col("asset_id").cum_count().over("time").alias("rank"))
        .join(daily.select("time", "quota"), on="time", how="left")
        .filter(pl.col("rank") <= pl.col("quota"))
        .select("time", "asset_id").sort("time", "asset_id")
    )


def _key_hash(keys: pl.DataFrame) -> str:
    # Audit hashing visits only the bounded sampled narrow keys, never wide rows.
    digest = hashlib.sha256()
    for day, asset in keys.iter_rows():
        digest.update(json.dumps([day.isoformat(), asset], separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()

@contextmanager
def capture_training_audits():
    """Collect immutable numerical lineage; the application persists it."""
    records = []
    token = _audits.set(records)
    try:
        yield records
    finally:
        _audits.reset(token)

def _register(function):
    operator = Operator(function, input_mode="composer", minimum_inputs=1, maximum_inputs=None, version="2",
        contract=OperationContract(execution=ExecutionMode.EAGER_BARRIER,
            density=InputDensity.DENSE_REQUIRED, trace_rule=TraceRule.PARENT_MAX))
    OPERATOR_REGISTRY.add(operator.registry_name, operator)
    return operator

def _rolling(features, labels, *, window, fit_every, min_samples, max_samples,
        label_maturity, anchor_offset, kind, fit, predict, lineage, restore,
        seed, estimator_parameters, audit_summary, label_end=None, label_available=None):
    if not features:
        raise ValueError("training requires at least one feature")
    for name, value in (("window",window),("fit_every",fit_every),("min_samples",min_samples),
            ("max_samples",max_samples),("label_maturity",label_maturity)):
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_samples < min_samples or anchor_offset < 0:
        raise ValueError("invalid sampling or anchor bounds")
    names = [f"feature_{index}" for index in range(len(features))]
    frame = features[0].select("time", "asset_id", (pl.col("value").is_not_null() & pl.col("value").is_finite()).alias(names[0]))
    for name, source in zip(names[1:], features[1:]):
        frame = frame.join(source.select("time", "asset_id", (pl.col("value").is_not_null() & pl.col("value").is_finite()).alias(name)), on=["time","asset_id"], how="left")
    frame = frame.join(labels.select("time","asset_id",pl.col("value").alias("label")), on=["time","asset_id"], how="left").sort(["time","asset_id"])
    dates = list(operator_execution_calendar()) or frame["time"].unique().sort().to_list()
    calendar_dates = set(dates)
    if dates != sorted(calendar_dates) or any(day not in calendar_dates for day in frame["time"].unique()):
        raise ValueError("training requires an ordered calendar covering every observation")
    index_frame = pl.DataFrame({"time": dates, "session": range(len(dates))})
    frame = frame.join(index_frame,on="time",how="left")
    valid_features = pl.all_horizontal([pl.col(name).fill_null(False) for name in names])
    for name, boundary in (("label_end", label_end), ("label_available", label_available)):
        if boundary is not None:
            frame = frame.join(boundary.select("time", "asset_id", pl.col("value").alias(name)), on=["time", "asset_id"], how="left")
    # Public boundary panels encode dates as Gregorian ordinals. Missing supplied
    # evidence is ineligible; absent panels retain the fixed-horizon contract.
    def wide(keys):
        selected = keys.select("time", "asset_id")
        first, last = keys["time"].min(), keys["time"].max()
        for name, source in zip(names, features):
            values = source.head(0) if first is None else source.filter(pl.col("time").is_between(first,last))
            selected = selected.join(values.select("time", "asset_id", pl.col("value").alias(name)), on=["time", "asset_id"], how="left")
        return selected
    saved = restored_operator_state()
    model = None if saved is None or saved["model"] is None else restore(saved["model"])
    continued = None if saved is None else date.fromisoformat(saved["through"])
    by_date = {key[0]:sample for key,sample in frame.partition_by("time",as_dict=True,maintain_order=True).items()}
    output = []
    for index, day in enumerate(dates):
        if continued is not None and day <= continued:
            output.append(by_date.get(day,frame.head(0)).select("time","asset_id").with_columns(pl.lit(None,dtype=pl.Float64).alias("value")))
            continue
        if (index + anchor_offset) % fit_every == 0:
            last = index - label_maturity
            first = max(0, last-window+1)
            eligible = pl.col("session").is_between(first,last) & valid_features & pl.col("label").is_not_null() & pl.col("label").is_finite()
            for boundary in ("label_end", "label_available"):
                if boundary in frame.columns:
                    eligible &= pl.col(boundary).is_not_null() & (pl.col(boundary) <= day.toordinal())
            candidates = frame.filter(eligible)
            selected = date_balanced_training_keys(candidates, budget=max_samples, seed=seed)
            training = wide(selected).join(candidates.select("time", "asset_id", "label"), on=["time", "asset_id"], how="left").sort("time", "asset_id")
            audit = {"model":kind,"fit_date":day,"training_start":training["time"].min(),
                "training_end":training["time"].max(),"sample_count":training.height,
                "feature_order":names,"label_maturity":label_maturity,
                "sampling_version":SAMPLING_VERSION, "seed":seed, "budget":max_samples,
                "training_key_hash":_key_hash(selected), "eligible_count":candidates.height,
                "requested_start": dates[first] if last >= 0 else None,
                "requested_end": dates[last] if last >= 0 else None,
                "daily_counts": [{"time": row["time"], "count": row["len"]} for row in selected.group_by("time").len().sort("time").to_dicts()],
                "estimator_parameters": estimator_parameters,
                "label_evidence": "row_boundaries" if label_end is not None and label_available is not None else "fixed_maturity_only"}
            admitted = candidates.join(selected, on=["time", "asset_id"], how="inner")
            for boundary in ("label_end", "label_available"):
                if boundary in admitted.columns:
                    upper = admitted[boundary].max()
                    audit[f"maximum_{boundary}"] = None if upper is None else date.fromordinal(int(upper))
            model = None
            if training.height >= min_samples:
                model = fit(training.select(names).to_numpy(),training["label"].to_numpy())
                audit.update(status="fitted", **audit_summary(model))
            else:
                audit.update(status="unavailable",reason="insufficient_mature_training_samples")
            collector = _audits.get()
            if collector is not None:
                collector.append(audit)
        current = by_date.get(day,frame.head(0))
        valid = wide(current.filter(valid_features))
        predictions = valid.select("time","asset_id").with_columns(pl.Series("value",predict(model,valid.select(names).to_numpy()))) if model is not None and valid.height else current.head(0).select("time","asset_id").with_columns(pl.lit(None,dtype=pl.Float64).alias("value"))
        output.append(current.select("time","asset_id").join(predictions,on=["time","asset_id"],how="left"))
    if dates:
        save_operator_state({"through":dates[-1].isoformat(),"model":None if model is None else lineage(model)})
    return pl.concat(output) if output else features[0].head(0)

@_register
def rolling_elastic_net_prediction(*features: pl.DataFrame, labels: pl.DataFrame,
        label_end: pl.DataFrame | None = None, label_available: pl.DataFrame | None = None,
        window: int = 252, fit_every: int = 20, min_samples: int = 200,
        max_samples: int = 100000, label_maturity: int = 2, anchor_offset: int = 0,
        alpha: float = 0.001, l1_ratio: float = 0.5, max_iter: int = 10000,
        tolerance: float = 1e-6, seed: int = 1729) -> pl.DataFrame:
    """Fit population RMS-scaled Elastic Net using only matured past labels."""
    names = tuple(f"feature_{index}" for index in range(len(features)))
    def fit(values,target):
        scaler = ZeroPreservingRmsScaler.fit(values,feature_names=names)
        model = fit_elastic_net(scaler.transform(values),target,alpha=alpha,l1_ratio=l1_ratio,max_iter=max_iter,tolerance=tolerance)
        if not model.converged:
            raise ValueError("Elastic Net failed to converge")
        return scaler,model
    return _rolling(features,labels,window=window,fit_every=fit_every,min_samples=min_samples,
        max_samples=max_samples,label_maturity=label_maturity,anchor_offset=anchor_offset,kind="elastic_net",fit=fit,
        seed=seed, estimator_parameters={"alpha":alpha,"l1_ratio":l1_ratio,"max_iter":max_iter,"tolerance":tolerance}, label_end=label_end, label_available=label_available,
        predict=lambda fitted,values:fitted[1].predict(fitted[0].transform(values)),
        lineage=lambda fitted:{"scaler":fitted[0].to_dict(),"estimator":fitted[1].to_dict()},
        audit_summary=lambda fitted:{"scaler":fitted[0].to_dict(),"estimator":fitted[1].to_dict()},
        restore=lambda state:(ZeroPreservingRmsScaler.from_dict(state["scaler"]),ElasticNetModel.from_dict(state["estimator"])))

@_register
def rolling_lightgbm_prediction(*features: pl.DataFrame, labels: pl.DataFrame,
        label_end: pl.DataFrame | None = None, label_available: pl.DataFrame | None = None,
        window: int = 252, fit_every: int = 20, min_samples: int = 200,
        max_samples: int = 100000, label_maturity: int = 2, anchor_offset: int = 0,
        num_leaves: int = 31, learning_rate: float = 0.05, min_data_in_leaf: int = 20,
        lambda_l1: float = 0.0, lambda_l2: float = 1.0, feature_fraction: float = 1.0,
        num_boost_round: int = 100, max_bin: int = 255, max_depth: int = -1,
        min_gain_to_split: float = 0.0, seed: int = 1729) -> pl.DataFrame:
    """Fit deterministic CPU LightGBM; hardware limits come from ResourceLimits."""
    candidate = LightGBMCandidate(num_leaves,learning_rate,min_data_in_leaf,lambda_l1,lambda_l2,feature_fraction)
    adapter = LightGBMEstimatorAdapter({"solver":{ "num_boost_round":num_boost_round,"max_bin":max_bin,
        "max_depth":max_depth,"min_gain_to_split":min_gain_to_split,"seed":seed},"search":{"candidates":[candidate.to_dict()]}})
    names = tuple(f"feature_{index}" for index in range(len(features)))
    return _rolling(features,labels,window=window,fit_every=fit_every,min_samples=min_samples,
        max_samples=max_samples,label_maturity=label_maturity,anchor_offset=anchor_offset,kind="lightgbm",
        seed=seed, estimator_parameters={**candidate.to_dict(),"num_boost_round":num_boost_round,"max_bin":max_bin,"max_depth":max_depth,"min_gain_to_split":min_gain_to_split,"seed":seed}, label_end=label_end, label_available=label_available,
        fit=lambda values,target:adapter.fit_candidate(values,target,feature_names=names,sample_weight=np.ones(len(target)),candidate=candidate),
        predict=adapter.predict,lineage=lambda fitted:{"estimator":adapter.checkpoint(fitted)},audit_summary=adapter.diagnostics,restore=lambda state:adapter.restore(state["estimator"]))
