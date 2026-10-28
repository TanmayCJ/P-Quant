# Pre-registration 2: contrarian LLM news signal with direction labels

Written 2026-10-24, **before** any LLM scores were computed for the assets below.
Frozen code: the `kronos-agent.zip` that first contains this file (keep a renamed copy on Drive).
Nothing in the prompt, model, headline selection, assets, dates or tests may change after
scoring starts. Results are reported whatever their sign or significance.

## Background (exploratory, already seen)
Pre-registration 1 (range elicitation scaled by volatility) failed to replicate; diagnostics and a
simulation showed that volatility-scaled ranges create a mechanical signal (Phase A). Using
direction labels instead (prompt: "will this news push the price up or down over the next 5
trading days? A: up / B: down / C: no clear effect"), the signal s = P(A) - P(B) on SPY + 8
large caps/ETFs (QQQ, IWM, AAPL, MSFT, NVDA, AMZN, TSLA, JPM; 2024-07..2026-10) gave a pooled
IC of -0.098 (t = -2.28), shuffled-news placebo +0.008 (t = 0.22), corr(s, volatility) = -0.05.

## Hypothesis
H1: s is negatively correlated with the realized return over the next 5 trading days.

## Confirmation set (none of these assets used before)
GOOGL, META, XOM, UNH, V, HD, PG, KO, LLY, AVGO, COST, WMT, BAC, CVX, MRK, PEP, ADBE, CRM,
NFLX, ORCL. Origins: every 5th trading day, 2024-07-01 to 2026-10-10 (same grid as before).
Information time: prior trading day 20:00 UTC. Headlines: Benzinga via Alpaca, created_at and
updated_at <= information time, 48h lookback, max 25, newest first.

## Fixed procedure
Llama-3.1-8B-Instruct, 4-bit NF4, letter-probability readout over A/B/C after "Answer:",
prompt as `kronos_agent/llm.py::build_label_messages`; config `llm.mode = labels`.
Realized return r = log(close[i+4] / close[i-1]) (5 trading days from the prior close).

## Primary test
Per asset, z-score s and r over that asset's origins; d = z(s) * z(r); average d across assets
per origin date; one-sided Newey-West t-test (lag 0) of mean(d) < 0. H1 confirmed if p < 0.05.

## Secondary (reported, not used for the decision)
1. Shuffled-news placebo (same assets, headlines permuted across dates): expected null.
2. Next-day horizon using the same scores: r1 = log(close[i] / close[i-1]); two-sided test
   (reversal vs. continuation, cf. Lopez-Lira & Tang).
3. Signal with volatility partialled out; per-asset ICs.
