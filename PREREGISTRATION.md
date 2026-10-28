# Pre-registration: contrarian LLM news signal (confirmation study)

Written 2026-10-23, **before** any confirmation-set LLM scores were computed.
Frozen code: the `kronos-agent.zip` that first contains this file. Nothing in the prompt,
model, headline selection, bins or tests below may change after confirmation scoring starts.

## Exploratory finding being tested
On SPY (5-trading-day horizon, origins 2024-07 to 2026-10), the expected return implied by
Llama-3.1-8B-Instruct's next-token distribution over 7 volatility-scaled return bins,
computed from leakage-safe Benzinga headlines, was negatively correlated with the realized
return: IC = -0.15 (HAC p = 0.021); anonymized prompts gave -0.16 (p = 0.011); shuffled-news
placebo gave -0.08 (p = 0.17). BTC (24h, 2025 H1) gave -0.19 (p = 0.16, n.s.).

## Hypothesis
H1: E_Q[r] (raw q, real headlines) is negatively correlated with the realized horizon return.

## Confirmation sets (none used in the exploratory analysis)
| Family | Assets | Horizon | Origins |
|---|---|---|---|
| Equities | QQQ, IWM, AAPL, MSFT, NVDA, AMZN, TSLA, JPM | 5 trading days | every 5th trading day, 2024-07-01 to 2026-10-10 |
| Crypto | BTC (2024-07-01..2024-12-31 and 2025-07-01..2026-10-15), ETH (2024-07-01..2026-10-15) | 24 hours | daily 00:00 UTC |

Equities are new assets over the same calendar period (cross-sectional replication);
BTC windows are new time periods, ETH is a new asset (temporal + cross-sectional).
All origins are after the Llama-3.1 (2023-12) and Kronos (2024-06) training cutoffs.

## Fixed procedure
- Model `meta-llama/Llama-3.1-8B-Instruct`, 4-bit NF4, letter-logit readout, prompt as in
  `kronos_agent/llm.py`; bins = 7, edges at (-1.5, -0.75, -0.25, 0.25, 0.75, 1.5) x sigma_h,
  sigma_h from the last 64 bars; headlines per `kronos_agent/news.py` (created_at and
  updated_at <= information time; 48h lookback for equities, 24h for crypto; max 25).
- Realized return r = log(close[i+h-1] / close[i-1]); E_Q[r] from `fusion.llm_only_samples`.

## Primary test (one per family, Holm-corrected at family-wise alpha = 0.05)
Per asset, z-score E_Q[r] and r over that asset's confirmation origins; d = z(E_Q) * z(r).
Average d across assets on each origin date (Fama-MacBeth style), then a one-sided
Newey-West t-test (lag 0; targets do not overlap) of mean(d) < 0.
H1 is confirmed for a family if its Holm-adjusted one-sided p < 0.05.

## Secondary (reported, not used for the decision)
Per-asset IC with HAC p; anonymized-prompt variant (expected: same sign); shuffled-news
placebo (expected: no effect, one-sided p > 0.05); pooled IC across families.

## Reporting commitment
Results are reported whatever their sign or significance, including failure to replicate.
