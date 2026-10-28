# Community audit runs

Each folder in `runs/` is one audit run contributed by the community: one model, on one asset,
at one setting, scored by this harness without changes.

```
runs/
  _template/                    copy this to start
  <handle>-<asset>-<model>/
    config.yaml                 the exact config you ran
    metrics.csv                 from reports/<tag>/metrics.csv
    tests.csv                   from reports/<tag>/tests.csv
    PROVENANCE.md               data source, dates, commit, hardware, changes
```

Results only: no prices, news, sampled paths or weights. See [CONTRIBUTING.md](../CONTRIBUTING.md)
for the full steps and the contribution terms.
