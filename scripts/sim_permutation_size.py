"""Size check of the pooled IC tests under the null: HAC vs permutation rejection rates at 5%
on simulated no-information panels (CPU).

python scripts/sim_permutation_size.py 200 2000    # n_panels, permutation draws
"""
import sys
import time

import numpy as np
import pandas as pd

from kronos_agent.artifact import _resid, simulate
from kronos_agent.confirm import family_test, perm_test

n_panels, B = int(sys.argv[1]), int(sys.argv[2])
rows, t0 = [], time.time()
for seed in range(n_panels):
    sim = simulate(info=0.0, seed=seed)
    for (mode, variant), f in sim.groupby(["mode", "variant"]):
        f = f.copy()
        f["eq_partial"] = np.nan
        for _, idx in f.groupby("asset").groups.items():
            g = f.loc[idx]
            f.loc[idx, "eq_partial"] = _resid(g["eq"].to_numpy(float), g["sigma"].to_numpy(float))
        for col in ("eq", "eq_partial"):
            hac = family_test(f.assign(eq=f[col]), variant)
            pt = perm_test(f, variant, B=B, seed=seed, col=col)
            rows.append({"mode": mode, "variant": variant, "signal": "raw" if col == "eq" else "partial",
                         "hac_rej_neg": hac["p_one_sided"] < 0.05, "perm_rej_neg": pt["p_one_sided_neg"] < 0.05,
                         "hac_rej_two": hac["p_one_sided"] < 0.025 or hac["p_one_sided"] > 0.975,
                         "perm_rej_two": pt["p_two_sided"] < 0.05})
r = pd.DataFrame(rows).groupby(["mode", "variant", "signal"]).mean()
print(f"{n_panels} null panels, B={B}, {time.time() - t0:.0f}s")
print(r.round(3).to_string())
