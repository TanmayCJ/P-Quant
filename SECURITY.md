# Security Policy

## Reporting a problem

Please **do not** open a public issue for security problems.

Email **tannycjain@gmail.com** with:

- what the problem is and where it is (file, function, config)
- how to reproduce it
- what impact you think it has

You should get a reply within 7 days. Once the problem is fixed, you will be credited in the
release notes unless you prefer not to be.

## What counts

- Code paths that could run untrusted code (for example loading pickles, configs or model files)
- Leaks of credentials, tokens or API keys
- Dependencies with known vulnerabilities that P-Quant pulls in
- **Leakage in the audit itself**: any bug that lets data from after a forecast's origin, or from
  inside a model's training window, reach the scoring. This breaks the point of the harness, so
  please report it here as well, or as a normal issue if it is not sensitive.

## Supported versions

Only the latest release on `main` gets fixes.

## Not financial advice

P-Quant is a research tool for evaluating forecasts. Nothing in this repository is investment
advice, and no output should be used to make trading decisions.
