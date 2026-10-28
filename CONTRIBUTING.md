# Contributing to P-Quant

Thanks for helping. P-Quant is an open audit layer: it checks whether a market forecasting model
(Kronos, Chronos, an LLM reading the news, or your own model) actually beats simple baselines once
leakage, calibration and multiple testing are handled properly.

There are three ways to contribute:

| Type | What you send | Where it goes |
|---|---|---|
| **Audit runs** | a config, the metrics the harness produced, and where the data came from | `contrib/runs/` |
| **Tests** | new unit tests or regression cases for the harness | `tests/` |
| **Code** | fixes, new baselines, new model adapters, new metrics | `kronos_agent/`, `scripts/` |

> [!IMPORTANT]
> Please read [Contribution terms](#contribution-terms) before opening a pull request.
> Opening a PR means you agree to them.

---

## Setup

```bash
git clone https://github.com/TanmayCJ/P-Quant.git
cd P-Quant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[data,stats]"                       # add model / chronos / llm extras as needed
python tests/run_all.py                              # all tests should pass before you change anything
```

Model sampling needs a GPU. The notebooks in `notebooks/` run on a free Colab T4.

---

## 1. Contributing an audit run

An audit run is one model, on one asset, at one setting, scored by this harness. Each accepted
run adds one more data point to the shared record of what these models can and cannot do.

**Steps**

1. Copy the template: `cp -r contrib/runs/_template contrib/runs/<your-handle>-<asset>-<model>`
2. Put your config in `config.yaml`. Start from one in `configs/` and change only what you need.
3. Run the harness unchanged:
   ```bash
   python scripts/run_sampling.py --config contrib/runs/<dir>/config.yaml --root <workdir> --kronos-path <Kronos clone>
   python scripts/run_eval.py     --config contrib/runs/<dir>/config.yaml --root <workdir>
   ```
   `--kronos-path` points at a clone of `shiyu-coder/Kronos` and is only needed for Kronos models.
4. Copy `<workdir>/reports/<tag>/metrics.csv` and `tests.csv` into your run folder.
5. Fill in `PROVENANCE.md`: data source, date range, harness commit (`git rev-parse HEAD`),
   hardware, and anything you changed.
6. Open a PR using the **Audit run** checklist.

**Rules for runs**

- **Results only, never raw data.** Do not commit prices, news text, sampled paths or model
  weights. `.gitignore` already blocks `data/`, `paths/`, `news/`, `reports/` and `*.parquet`.
  Small CSVs of summary metrics are fine.
- **Freely redistributable sources only.** No Bloomberg, Refinitiv, paid exchange feeds, or any
  source whose terms forbid sharing derived results.
- **Respect the training cutoff.** Results only count if the scored window starts after the
  model's published training cutoff. Say which cutoff you used.
- **Report what you got.** Negative and null results are as welcome as positive ones. Do not tune
  settings on the scoring window and then submit the best run.
- **Pre-register if you can.** If you are claiming an effect, write the hypothesis and test
  before running, in the style of [`PREREGISTRATION.md`](PREREGISTRATION.md), and commit it first.

---

## 2. Contributing tests

- Tests live in `tests/` and run with `python tests/run_all.py`.
- Each test should be deterministic (fixed seeds) and need no network access or GPU.
- Use synthetic data (random walks, known-answer cases) instead of downloaded prices.
- A bug fix should come with a test that fails before the fix and passes after it.

---

## 3. Contributing code

- Keep changes small and focused: one fix or feature per PR.
- Follow the style of the surrounding code: plain functions, numpy/pandas, configs in YAML.
- New analyses go behind a config option, so existing configs keep producing the same numbers.
- Do not change the scoring of existing metrics without opening an issue first. Published results
  depend on them.
- Run `python tests/run_all.py` before opening the PR.

### Commit sign-off

Sign off every commit to certify you wrote it or have the right to submit it, under the
[Developer Certificate of Origin](https://developercertificate.org/):

```bash
git commit -s -m "Add Chronos-bolt adapter"
```

---

## Contribution terms

By submitting a contribution (code, tests, configs, run results, documentation or anything else)
you agree that:

1. **License.** Your contribution is licensed under the [MIT License](LICENSE), the same as the
   rest of the repository.
2. **Use for model development.** The maintainer may use any contribution, including audit-run
   results, configs and tests, to evaluate, train and develop other models, tools and products,
   including ones that are not open source and are not part of this repository. The open-source
   scope of P-Quant is the audit layer in this repository.
3. **Your rights.** You wrote the contribution yourself or have the right to submit it, and it
   does not include data or code whose license or terms of service forbid this use.
4. **No personal data.** Contributions contain no personal information about anyone.
5. **No obligation.** The maintainer may decline, edit or remove any contribution.

If you cannot agree to these terms, please open an issue to discuss it before contributing.

---

## Questions

Open a [discussion or issue](https://github.com/TanmayCJ/P-Quant/issues). For security problems,
see [`SECURITY.md`](SECURITY.md). Everyone taking part follows the
[Code of Conduct](CODE_OF_CONDUCT.md).
