from datetime import date, datetime, time, timedelta

import polars as pl
import pytest

from bagelquant_core import CoreStore, Domain, Node
from bagelquant_core.materialization import MaterializationKey, NodeMaterialization


@pytest.mark.parametrize("dynamic", [False, True])
def test_persisted_canonical_date_window_pushes_into_scan_and_preserves_values_traces(tmp_path, dynamic):
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(96)]
    members = pl.DataFrame({"time": [day for day in dates for _ in range(2)],
                            "asset_id": ["A", "B"] * 96,
                            "active": [index % 7 != 0 for index in range(192)]})
    domain = Domain(calendar=dates, universe=members if dynamic else ["A", "B"])
    frame = members.select("time", "asset_id").with_columns(
        pl.Series("value", [None if index % 5 == 0 else float("nan") if index % 11 == 0 else float(index)
                            for index in range(192)]),
        (pl.col("time") + pl.duration(days=1)).alias("available_date"),
    ).filter(~((pl.col("asset_id") == "A") & (pl.col("time") == dates[50])))
    reference = domain.apply_membership(frame.with_columns(pl.col("value").fill_nan(None))).sort("time", "asset_id")
    panel = Node.from_domain(frame, domain, identity="original-input", trace_columns=("available_date",))
    key = MaterializationKey("node", "kernel", ("original-input",), domain.signature, "unchanged-context")
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    store.initialize()
    store.publish(NodeMaterialization(key, panel))
    saved = store.lookup(key)
    assert saved is not None and saved.key.identity == key.identity
    plan = saved.panel.lazy(include_traces=True).filter(pl.col("time").is_between(dates[40], dates[71]))
    # Pushdown into the source is the regression: a post-scan FILTER still
    # decompresses historical row groups for every canonical 32-session block.
    assert "SELECTION:" in plan.explain()
    assert plan.collect().equals(reference.filter(pl.col("time").is_between(dates[40], dates[71])))
    assert saved.panel.collect(dense=False, include_traces=True).equals(reference)
    assert saved.panel.trace_columns == ("available_date",)
    assert saved.panel.value_type == panel.value_type


@pytest.mark.parametrize("dtype", [pl.String, pl.Datetime("us")])
@pytest.mark.parametrize("lazy", [False, True])
def test_noncanonical_string_and_datetime_conversion_still_preserves_typed_boundary(dtype, lazy):
    dates = [date(2020, 1, index) for index in (1, 2, 3)]
    times = [day.isoformat() for day in dates] if dtype == pl.String else [datetime.combine(day, time(12, 30)) for day in dates]
    frame = pl.DataFrame({"time": pl.Series(times, dtype=dtype), "asset_id": ["A"] * 3,
                          "value": [1., None, float("nan")], "available_date": dates})
    domain = Domain(calendar=dates, universe=["A"])
    panel = Node.from_domain(frame.lazy() if lazy else frame, domain, identity="original-input",
                             trace_columns=("available_date",))
    actual = panel.collect(dense=False, include_traces=True)
    assert actual.schema["time"] == pl.Date
    assert actual["time"].to_list() == dates
    assert actual["value"].to_list() == [1., None, None]
    assert actual["available_date"].to_list() == dates
