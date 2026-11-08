# Y-Map data

Use `Dataset/YMap` as the single data root. It contains the object models, SQL
catalog, measured labels, and source metadata. Measurements from different
workloads and environments are registered as separate scenarios.

```sh
ymap paper check --dataset native
ymap paper check --dataset archive
```

See the [dataset catalog](YMap/README.md) and
[provenance guide](../docs/DATA_AND_PROVENANCE.md) for file layout, counts,
attribution, and license terms. See the [manuscript workflow](../docs/PAPER_WORKFLOW.md)
to register an additional documented profiling run under this root.
