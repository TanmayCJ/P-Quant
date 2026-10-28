## What this PR does

<!-- One or two sentences. Link the issue if there is one: Closes #... -->

## Type

- [ ] Audit run (`contrib/runs/`)
- [ ] Tests
- [ ] Bug fix
- [ ] New feature (baseline, adapter, metric)
- [ ] Docs

## Checklist

- [ ] `python tests/run_all.py` passes
- [ ] Commits are signed off (`git commit -s`)
- [ ] No raw data, news text, sampled paths or model weights are committed
- [ ] Existing configs produce the same numbers (or the change is explained below)
- [ ] I have read and agree to the [contribution terms](../CONTRIBUTING.md#contribution-terms)

### For audit runs

- [ ] `config.yaml`, `metrics.csv`, `tests.csv` and `PROVENANCE.md` are all in the run folder
- [ ] The scored window starts after the model's training cutoff
- [ ] The data source allows sharing derived results
- [ ] Settings were not tuned on the scoring window

## Notes for the reviewer

<!-- Anything unusual, open questions, or results worth a closer look. -->
