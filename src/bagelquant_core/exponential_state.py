"""Bounded exponential moments with the Polars adjust/null/bias conventions."""
from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date

import polars as pl

from bagelquant_core.operator_state import checkpoint_capture_enabled, restored_operator_state, save_operator_state


def valid_ewm_state(state):
    assets = state.get("assets")
    if not isinstance(assets, Mapping):
        return False
    for asset, value in assets.items():
        if not isinstance(asset, str) or not isinstance(value, Mapping):
            return False
        observations = value.get("observations")
        if isinstance(observations, bool) or not isinstance(observations, int) or observations < 0:
            return False
        for name in ("cov", "weight", "sum_weight", "sum_squared_weight"):
            number = value.get(name)
            if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
                return False
        mean = value.get("mean")
        if mean is not None and (isinstance(mean, bool) or not isinstance(mean, (float, int)) or not math.isfinite(mean)):
            return False
    return True


def checkpoint_ewm(frame, *, alpha, min_periods, adjust, ignore_na, moment, bias=False):
    if not checkpoint_capture_enabled():
        return None
    # Use the numerical backend's parameter validation even for empty inputs.
    pl.Series([], dtype=pl.Float64).ewm_mean(alpha=alpha, min_samples=min_periods,
        adjust=adjust, ignore_nulls=ignore_na)
    restored = restored_operator_state()
    states = {} if restored is None else {asset: dict(state) for asset, state in restored['assets'].items()}
    through = None if restored is None else date.fromisoformat(restored['through'])
    output = []
    ordered = frame.sort(['asset_id','time'])
    for day, asset, value in ordered.select('time','asset_id','value').iter_rows():
        if through is not None and day <= through:
            output.append(None)
            continue
        valid = value is not None and not math.isnan(value)
        state = states.setdefault(asset, {'mean':None, 'cov':0.0, 'weight':1.0,
            'sum_weight':1.0, 'sum_squared_weight':1.0, 'observations':0})
        if valid:
            state['observations'] += 1
        mean = state['mean']
        if mean is None:
            if valid:
                state['mean'] = value
        elif valid or not ignore_na:
            decay = 1.0-alpha
            state['weight'] *= decay
            state['sum_weight'] *= decay
            state['sum_squared_weight'] *= decay*decay
            if valid:
                new_weight = 1.0 if adjust else alpha
                total = state['weight'] + new_weight
                updated = mean if mean == value else (state['weight']*mean + new_weight*value)/total
                state['cov'] = (state['weight']*(state['cov']+(mean-updated)**2)
                    + new_weight*(value-updated)**2)/total
                state['mean'] = updated
                state['weight'] = total
                state['sum_weight'] += new_weight
                state['sum_squared_weight'] += new_weight*new_weight
                if not adjust:
                    state['sum_weight'] /= total
                    state['sum_squared_weight'] /= total*total
                    state['weight'] = 1.0
        result = None
        if valid and state['observations'] >= max(1,min_periods):
            if moment == 'mean':
                result = state['mean']
            elif bias:
                result = state['cov']
            else:
                square = state['sum_weight']**2
                denominator = square-state['sum_squared_weight']
                result = square/denominator*state['cov'] if denominator > 0 else 0.0
            if moment == 'std' and result is not None:
                result = math.sqrt(max(0.0,result))
        output.append(result)
    if ordered.height:
        save_operator_state({'through':ordered['time'].max().isoformat(), 'assets':states})
    return ordered.with_columns(pl.Series('value',output,dtype=pl.Float64))
