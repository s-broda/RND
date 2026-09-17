# Closed-form risk-neutral densities from option prices

Replication package for

> Simon A. Broda, *A New Closed-Form Estimator for the Risk-Neutral Density Implied by Option Prices*.

The headline estimator is `python/rnd.py` (`estimate_rnd`). It rescales European calls to a complementary cdf, completes the unquoted tails, places masses at cell centres, and applies a Gaussian kernel with Schucany–Sommers twicing.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 python/replicate.py
```

This prints the Heston ISE table and the listed SPX/NDX pricing table, and writes `figures/ccdf_heston_rnd.pdf` and `figures/ccdf_listed.pdf`.

Compile the paper with `make` (TeX Live with `elsarticle`).

## Data

Cboe delayed quotes cannot be redistributed as a full option-chain dump. The three **working slices** used in the paper (strikes, OTM mids, parity-filled other side, and the snapshot metadata) are in `python/results/`:

- `spx_20261218.csv` — SPX 18 December 2026, snapshot 6 September 2026
- `spx_20270319.csv` — SPX 19 March 2027, same snapshot
- `ndx_20261218.csv` — NDX 18 December 2026, snapshot 8 September 2026

Source: Cboe delayed quotes, `https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json` (and `_NDX.json`). The slices are provided solely so the tables and figures in the paper can be reproduced.

## License

Code is MIT (see `LICENSE`). Market data remain Cboe’s; the CSV slices are a research extract for replicating this paper.
