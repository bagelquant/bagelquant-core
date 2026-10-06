"""Standalone DSL, shared graph, durable cache and restart using synthetic data."""
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import polars as pl
from bagelquant_core import CoreStore, Domain, Graph, Node


def main():
    with TemporaryDirectory(prefix="bagelquant-core-") as temporary:
        root = Path(temporary)
        store = CoreStore(root / "meta.sqlite", root / "artifacts")
        days = [date(2024, 1, 2), date(2024, 1, 3)]
        domain = Domain(calendar=days, universe=["A", "B"])
        data = domain.grid_lazy().with_columns(pl.when(pl.col("asset_id") == "A").then(1.).otherwise(3.).alias("value")).collect()
        x = Node.from_domain(data, domain, source_key="close")
        graph = Graph(store=store, graph_id="research")
        merged = graph.add_dsl("output = zscore(close)", inputs={"close": x})
        graph.register_context("universe", anchor=days[0], information_cutoff=days[-1])
        for run in range(2):
            graph = Graph(store=CoreStore(root / "meta.sqlite", root / "artifacts"), graph_id="research")
            plan = graph.plan_update({"universe": {"close": x}}, through=days[-1])
            while not plan.complete:
                for node_id in plan.ready("universe"):
                    plan.execute("universe", node_id)
            receipt = plan.publish()
            node_id = next(iter(merged.outputs.values()))
            print(f"run {run + 1}", store.read_values(receipt["results"]["universe"][node_id]))


if __name__ == "__main__":
    main()
