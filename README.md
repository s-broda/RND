# Thricing: A Sixth-Order Kernel Estimator of the Risk-Neutral Density from a Single Option Chain

Replication code for

> Simon A. Broda, *Thricing: A Sixth-Order Kernel Estimator of the Risk-Neutral Density from a Single Option Chain*.

The headline estimator rescales European calls to a complementary cdf, fits that measure with a smoothing spline, completes the unquoted tails, and applies a Gaussian convolution with the three-bandwidth jackknife \(\tfrac83 f_h-2f_{h\sqrt{2}}+\tfrac13 f_{2h}\). `estimate_rnd` in `python/rnd.py` is that estimator. `method` selects the convolution. `naive` sums the closed-form cell integrals. `fft` is one real FFT of the sampled spline. `fast`, the default, is one FFT of the order-eight box moments in `python/density_closed.c`, with thricing as one multiplier. `estimate_rnd` compiles that file when a C compiler is present. If it cannot be compiled, `fast` uses the sampled-spline FFT. `fft` and `naive` do not use the compiler. The penalty is zero when the thriced interpolant scores at most the kernel's own positive-part variation, about \(1.007\), on \([0.60F, 1.30F]\). The tables report that same functional. The bandwidth is \(0.34\gamma F \sigma_{\mathrm{ATM}} \sqrt{T}\, n^{-1/9}\). The factor \(\gamma\) is the interquartile range of the positive call butterflies, divided by the normal interquartile range of scale \(F \sigma_{\mathrm{ATM}} \sqrt{T}\), and clipped to \([0.90, 1.10]\). The defaults are thricing, tails, the `n^{-1/9}` bandwidth, `penalty="closed"`, `tv_cap` equal to that kernel score, and `method="fast"`. `penalty="closed"` is `((t - tv_cap)_+ / (v_* - 1)) h^4 / (δ F^3)`, with `t` the clipped variation of the thriced interpolant on `[0.60F, 1.30F]`. `penalty="search"` keeps the 17-point grid and the curvature corner. Pass `c=0` for the cubic through the knots, `kernel="twice"` or `kernel="plain"` for the other convolutions, `tails=False` to drop the endpoint knots, and `h` or `deriv_c` to change the bandwidth.

## Run (paper tables and figures)

```bash
python3 -m pip install -r requirements.txt
python3 python/replicate.py exact      # Tables 1-2: exact quotes
python3 python/replicate.py mc         # Tables 3-5: Monte Carlo (several hours on two cores)
python3 python/replicate.py sens       # Table 6: sensitivity to the bandwidth constant and penalty
python3 python/replicate.py listed     # Tables 7-8, B.1: the four listed chains
python3 python/replicate.py extra      # Table 9, C.1: every further monthly chain (needs the Cboe dumps)
python3 python/replicate.py figures    # figures/ccdf_listed.pdf
python3 python/tables.py     # writes tables/*.tex from python/results/tables/*.json
```

`replicate.py` tunes every method by the same even/odd pricing cross-validation on a common geometric
bandwidth grid, nested inside the hold-out, and also reports the rules of the source papers.
Errors are reported in index points, in quoted half-spreads (the slices carry the bid-ask spread of
the quoted side), and in implied volatility. The Monte Carlo covers Heston, variance gamma and the
Bates jump diffusion (`src/bates.py`) at three noise levels on three meshes, plus an oracle-bandwidth
run (`replicate.py oracle`) and the competitors at our bandwidth (`replicate.py transfer`).
`variants.py` runs the GCV, positive-part and weighted variants; `mc_driver.py` is the Monte Carlo
driver used for the paper. The Heston and variance-gamma figures of the exact designs are drawn by
`replicate.py figures` as well. The further chains of the appendix need the Cboe dumps
`python/results/cboe_*.json`, which are not shipped (see below).

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
