# Identifying Risk-Neutral Densities from an Empirical Measure of Listed Quotes

Replication code for

> Simon A. Broda, *Identifying Risk-Neutral Densities from an Empirical Measure of Listed Quotes*.

The headline estimator rescales European calls to a complementary cdf, fits that measure with a smoothing spline, completes the unquoted tails, and applies a Gaussian convolution with the three-bandwidth jackknife \(\tfrac83 f_h-2f_{h\sqrt{2}}+\tfrac13 f_{2h}\). The convolution is closed form. The penalty is zero when the thriced density is already unimodal. The bandwidth is \(0.38 F \sigma_{\mathrm{ATM}} \sqrt{T}\, n^{-1/9}\). `estimate_rnd` in `python/rnd.py` is the interpolating case of that convolution. Passing `h="mesh"` uses 1.2 times the median strike gap.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 python/replicate.py
```

This prints the Heston and variance-gamma ISE tables, the same competitors on those chains and on noisy Heston quotes, and the listed SPX/NDX/RUT pricing table, and writes `figures/ccdf_heston_rnd.pdf`, `figures/ccdf_vg_rnd.pdf`, `figures/ccdf_listed.pdf`, and `figures/ccdf_listed_kern.pdf`.

Every listed fit starts from the same unpenalized decreasing-convex projection of the quoted calls and is scored against the quoted mids. Aït-Sahalia–Duarte is one Gaussian local linear at the Fan–Gijbels plug-in, and the density is the scaled and shifted derivative of that fit. Bondarenko’s positive convolution approximation is a Gaussian kernel on the spot, mixed on a uniform strike grid, with nonnegative weights of unit mass and mean equal to the forward; the bandwidth minimises even/odd pricing error. Aït-Sahalia–Lo is Nadaraya–Watson of implied volatility on moneyness at the Appendix A bandwidth \(c\,s(K/F)\,n^{-1/9}\), with the Table II constant. The Grith–Härdle–Schienle bandwidth is the leave-one-out width of the local cubic, and the comparison also reports twice that width. The spline prices from the interpolant at \(0.38 F \sigma_{\mathrm{ATM}} \sqrt{T}\, n^{-1/9}\). Exact Heston and variance-gamma quotes are scored for the spline at the mesh rule and at the \(n^{-1/9}\) rule, and for the same competitors, on a dense grid and on a 32-strike grid. Noisy Heston repeats that comparison over 30 draws of implied-volatility noise with standard deviation 0.01. The reported estimator is that smoothing spline. The script prints it on the exact chains, on the noisy Heston draws, and as the listed row. The four working slices are one Cboe session, 23 September 2026.

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
