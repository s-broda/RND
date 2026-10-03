# Thricing Is All You Need

Replication code for

> Simon A. Broda, *Thricing Is All You Need: Identifying Risk-Neutral Densities from an Empirical Measure of Listed Quotes*.

The headline estimator rescales European calls to a complementary cdf, fits that measure with a smoothing spline, completes the unquoted tails, and applies a Gaussian convolution with the three-bandwidth jackknife \(\tfrac83 f_h-2f_{h\sqrt{2}}+\tfrac13 f_{2h}\). `estimate_rnd` in `python/rnd.py` is that estimator. `method` selects the convolution. `naive` sums the closed-form cell integrals. `fft` is one real FFT of the sampled spline. `fast`, the default, is one FFT of the order-eight box moments in `python/density_closed.c`, with thricing as one multiplier. `estimate_rnd` compiles that file when a C compiler is present. If it cannot be compiled, `fast` uses the sampled-spline FFT. `fft` and `naive` do not use the compiler. The penalty is zero when the thriced density scores at most the kernel's own positive-part variation, about \(1.007\). The bandwidth is \(0.34\gamma F \sigma_{\mathrm{ATM}} \sqrt{T}\, n^{-1/9}\). The factor \(\gamma\) is the interquartile range of the positive call butterflies, divided by the normal interquartile range of scale \(F \sigma_{\mathrm{ATM}} \sqrt{T}\), and clipped to \([0.90, 1.10]\). The defaults are thricing, tails, the `n^{-1/9}` bandwidth, `penalty="closed"`, `tv_cap` equal to that kernel score, and `method="fast"`. `penalty="closed"` is `((t - tv_cap)_+ / (v_* - 1)) h^4 / (δ F^3)`, read from the thriced interpolant. `penalty="search"` keeps the 17-point grid and the curvature corner. Pass `c=0` for the cubic through the knots, `kernel="twice"` or `kernel="plain"` for the other convolutions, `tails=False` to drop the endpoint knots, and `h` or `deriv_c` to change the bandwidth.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 python/replicate.py
```

This prints the Heston and variance-gamma ISE tables, the same competitors on those chains and on noisy Heston quotes, and the listed SPX/NDX/RUT pricing table, and writes `figures/ccdf_heston_rnd.pdf`, `figures/ccdf_vg_rnd.pdf`, `figures/ccdf_listed.pdf`, and `figures/ccdf_listed_kern.pdf`.

Every listed fit starts from the same unpenalized decreasing-convex projection of the quoted calls and is scored against the quoted mids. Aït-Sahalia–Duarte is one Gaussian local linear at the Fan–Gijbels plug-in, and the density is the scaled and shifted derivative of that fit. Bondarenko’s positive convolution approximation is a Gaussian kernel on the spot, mixed on a uniform strike grid, with nonnegative weights of unit mass and mean equal to the forward; the bandwidth minimises even/odd pricing error. Aït-Sahalia–Lo is Nadaraya–Watson of implied volatility on moneyness. The listed, exact, and noisy rows use the smallest moneyness bandwidth on a grid from 0.01 to 0.40 that leaves one peak and total variation at most 1.05, not the Appendix A constant. The Grith–Härdle–Schienle bandwidth is the leave-one-out width of the local cubic, and the comparison also reports twice that width. The spline is priced at the bandwidth \(0.34\gamma F \sigma_{\mathrm{ATM}} \sqrt{T}\, n^{-1/9}\), with \(\gamma\) the clipped butterfly factor above. Exact Heston and variance-gamma quotes are scored for the spline at an oracle bandwidth and at the \(n^{-1/9}\) rule, and for the same competitors, on a dense grid and on a 32-strike grid. Noisy Heston repeats that comparison over 30 draws of implied-volatility noise with standard deviation 0.01. The reported estimator is that smoothing spline. On each of the four listed chains that row has the smallest hold-out, and the same ranking holds in the average. The row has one peak, mass 1.0, and total variation 1.0. The script prints that row on the exact chains, on the noisy Heston draws, and as the listed row, and then the wall-clock time of `naive`, `fft`, and `fast` at penalty zero, with the penalty rule not applied. The twenty-three further chains in the appendix are not in this run. Under the Cboe Global Markets North American Data Policies, effective 1 September 2026, a recipient of delayed Cboe options quotes may not redistribute them externally except to a named affiliate, and historical quotes may be redistributed to anyone else only under a data agreement and Cboe's approval. The parenthetical is that time in seconds, half-up to three decimals. Those times are the computational section of the paper, where `fast` is the FFT+ME column. The four working slices are one Cboe session, 23 September 2026.

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
