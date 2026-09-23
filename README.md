# Identifying Risk-Neutral Densities from an Empirical Measure of Listed Quotes

Replication code for

> Simon A. Broda, *Identifying Risk-Neutral Densities from an Empirical Measure of Listed Quotes*.

The headline estimator is `python/rnd.py` (`estimate_rnd`). It rescales European calls to a complementary cdf, completes the unquoted tails, places masses at cell centers, and applies a Gaussian kernel with Schucany–Sommers twicing.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 python/replicate.py
```

This prints the Heston and variance-gamma ISE tables and the listed SPX/NDX/RUT pricing table, and writes `figures/ccdf_heston_rnd.pdf`, `figures/ccdf_vg_rnd.pdf`, and `figures/ccdf_listed.pdf`.

## Version 2

`paper_v2.tex` is a separate manuscript. It keeps the estimator and the exact-quote ablation, records that the signed density integrates to zero with first moment equal to the forward, and rescores the benchmarks. `python/replicate.py` is unchanged as an entry point. The v2 script is

```bash
python3 python/replicate_v2.py
```

It writes `figures_v2/` and `python/results/v2_summary.json`. The positive-convolution centers cover the quoted log-strike range and the fitted weights are not rescaled. The Yatchew–Härdle penalty, the Priestley–Chao bandwidth, and the Aït-Sahalia–Duarte pricing bandwidth are chosen by even/odd cross-validation. The noise study adds one volatility point of Gaussian noise on an irregular strike grid.

## Data

Cboe delayed quotes cannot be redistributed as a full option-chain dump. The **working slices** used in the paper (strikes, OTM mids, parity-filled other side, and the snapshot metadata) are in `python/results/`:

- `spx_20261218.csv` — SPX 18 December 2026, snapshot 6 September 2026
- `spx_20270319.csv` — SPX 19 March 2027, same snapshot
- `ndx_20261218.csv` — NDX 18 December 2026, snapshot 8 September 2026
- `rut_20261218.csv` — RUT 18 December 2026, snapshot 19 September 2026

Source: Cboe delayed quotes, `https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json` (and `_NDX.json`, `_RUT.json`). The slices are provided solely so the tables and figures in the paper can be reproduced.

## License

Code is MIT (see `LICENSE`). Market data remain Cboe’s; the CSV slices are a research extract for replicating this paper.

## Updating GitHub

The local `main` branch tracks `git@github.com:s-broda/RND.git`. A post-commit hook pushes each commit. First-time publish (requires a one-time `gh auth login` in the browser):

```bash
sh scripts/publish.sh
```
