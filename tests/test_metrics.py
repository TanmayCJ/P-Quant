import math
import tempfile
from pathlib import Path

import numpy as np

from kronos_agent import metrics as M


def test_crps_point_equals_abs_error():
    x = np.full((3, 10), 2.0)
    y = np.array([1.0, 2.0, 5.0])
    assert np.allclose(M.crps_ensemble(x, y), [1.0, 0.0, 3.0])


def test_crps_uniform_weights_match_unweighted():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(5, 32)), rng.normal(size=5)
    assert np.allclose(M.crps_ensemble(x, y), M.crps_ensemble(x, y, np.ones_like(x)))


def test_crps_gaussian_closed_form():
    sigma = 2.0
    x = np.random.default_rng(1).normal(0, sigma, size=(1, 4000))
    expected = sigma * (2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))  # y = 0
    assert abs(M.crps_ensemble(x, 0.0)[0] - expected) < 0.02


def test_weights_shift_quantiles():
    x = np.array([[0.0, 1.0, 2.0, 3.0]])
    w = np.array([[0.0, 0.0, 0.0, 1.0]])
    assert np.isclose(M.weighted_quantile(x, [0.5], w)[0, 0], 3.0)
    assert np.isclose(M.weighted_quantile(x, [0.5])[0, 0], 1.5)


def test_interval_coverage_calibrated():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3000, 200))
    y = rng.normal(size=3000)
    assert abs(M.interval_metrics(x, y, 0.8)["coverage"] - 0.8) < 0.03


def test_pit_uniform_when_calibrated():
    rng = np.random.default_rng(3)
    p = M.pit(rng.normal(size=(3000, 200)), rng.normal(size=3000))
    assert abs(p.mean() - 0.5) < 0.02 and abs((p < 0.1).mean() - 0.1) < 0.02


def test_rank_ic_and_direction():
    a = np.arange(10.0)
    assert np.isclose(M.rank_ic(a, a ** 3), 1.0)
    assert M.directional_accuracy([1, -1, 1], [2, -3, -1]) == 2 / 3


def test_dm_detects_worse_model():
    rng = np.random.default_rng(4)
    good = np.abs(rng.normal(0, 1.0, 500))
    bad = np.abs(rng.normal(0, 1.5, 500))
    r = M.dm_test(good, bad, h=1)
    assert r["stat"] < -3 and r["p"] < 0.01
    assert M.dm_test(good, good + 0.0, h=3)["mean_diff"] == 0


def test_block_bootstrap_ci_contains_mean():
    x = np.random.default_rng(5).normal(1.0, 1.0, 400)
    lo, hi = M.block_bootstrap_ci(x, block=10)
    assert lo < 1.0 < hi


def test_config_overrides_and_tag():
    from kronos_agent.config import load_config, run_tag

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text('asset: X\nsymbol: X\ntimeframe: 1h\ndata_start: "2020-01-01"\n'
                     'data_end: "2021-01-01"\nwindows: {a: ["2020-06-01", "2020-07-01"]}\n')
        base = load_config(p)
        cfg = load_config(p, ["amp=true", "n_paths=32", "calib.warmup=10"])
        assert cfg["amp"] is True and cfg["n_paths"] == 32
        assert cfg["calib"]["warmup"] == 10 and cfg["calib"]["window"] == 250
        assert run_tag(cfg) != run_tag(base) and run_tag(cfg).endswith("_fp16")
