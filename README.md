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

Every listed fit starts from the same unpenalized decreasing-convex projection of the quoted calls, the step shared by Yatchew–Härdle and Aït-Sahalia–Duarte, and is scored against the quoted mids. The listed comparison then chooses the Yatchew–Härdle penalty, the Aït-Sahalia–Duarte pricing bandwidth, and the positive-convolution width by even/odd out-of-the-money error. The kernel prices from the interpolant at the density-scale bandwidth, Silverman's rule. Priestley–Chao is the cubic convolved at that same bandwidth. Exact Heston and variance-gamma quotes are scored only for the kernel. The four working slices are one Cboe session, 23 September 2026.

## Data

Cboe delayed quotes cannot be redistributed as a full option-chain dump. The **working slices** used in the paper (strikes, OTM mids, parity-filled other side, and the snapshot metadata) are in `python/results/`:

- `spx_20261218.csv` — SPX 18 December 2026, snapshot 23 September 2026
- `spx_20270319.csv` — SPX 19 March 2027, same snapshot
- `ndx_20261218.csv` — NDX 18 December 2026, same date
- `rut_20261218.csv` — RUT 18 December 2026, same date

Source: Cboe delayed quotes, `https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json` (and `_NDX.json`, `_RUT.json`). The slices are provided solely so the tables and figures in the paper can be reproduced.

## License

Code is MIT (see `LICENSE`). Market data remain Cboe’s; the CSV slices are a research extract for replicating this paper.

## Updating GitHub

The local `main` branch tracks `git@github.com:s-broda/RND.git`. A post-commit hook pushes each commit. First-time publish (requires a one-time `gh auth login` in the browser):

```bash
sh scripts/publish.sh
```
