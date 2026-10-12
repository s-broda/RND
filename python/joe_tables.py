"""LaTeX tables from the JSON written by ``joe.py``. Writes ``tables/*.tex``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PY = Path(__file__).resolve().parent
ROOT = PY.parent
sys.path.insert(0, str(PY))
from joe import MAIN, METHODS, RULE  # noqa: E402

OUT = ROOT / "python" / "results" / "joe"
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)

LABEL = {k: v[2] for k, v in METHODS.items()}
SHORT = {"ours_rule": "Ours, rule", "ours_cv": "Ours, CV", "pca_cv": "PCA", "asl_cv": "AS--Lo",
         "asd_cv": "AS--D", "ghs_iv_cv": "GHS IV", "ghs_call_cv": "GHS call",
         "asl_app": "AS--Lo, A", "asd_plug": "AS--D, plug", "ghs_iv_loo": "GHS IV, LOO",
         "ghs_iv_loo2": "GHS IV, 2LOO", "ghs_call_loo": "GHS call, LOO", "ghs_call_loo2": "GHS call, 2LOO"}
MODEL = {"heston": "Heston", "vg": "VG", "bates": "Bates"}
CHAIN = {"dense": "dense", "sparse": "sparse", "listed": "listed"}
NESTED = {"quoted": "Quoted spline", "tails": "Tails", "twice": "Tails + twice", "thrice": "Tails + thrice"}


def _load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _sci(x):
    if x is None or not np.isfinite(x):
        return "{--}"
    m, e = f"{x:.2e}".split("e")
    return f"{m}e{int(e)}"


def _f(x, d=2):
    return "{--}" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _bold_min(values, fmt):
    """Format a list, bolding the minimum. ``values`` may hold None."""
    arr = np.array([np.inf if v is None or not np.isfinite(v) else v for v in values], float)
    j = int(np.argmin(arr)) if np.isfinite(arr).any() else -1
    out = []
    for i, v in enumerate(values):
        s = fmt(v)
        out.append(f"\\textbf{{{s}}}" if i == j else s)
    return out


def write(name, text):
    (TAB / f"{name}.tex").write_text(text)
    print("wrote", name)


# ---------------------------------------------------------------- exact


def exact_tables():
    ex = _load("exact")
    if ex is None:
        return
    rows = []
    for model in ("heston", "vg", "bates"):
        for chain in ("dense", "sparse"):
            rec = ex[f"{model}/{chain}"]
            for i, key in enumerate(("quoted", "tails", "twice", "thrice")):
                r = rec[key]
                first = f"{MODEL[model]}, {chain}" if i == 0 else ""
                rows.append(
                    f"{first} & {NESTED[key]} & {r['h_star']:.2f} & {_sci(r['ise_star'])} & "
                    f"{r['h_9']:.2f} & {_sci(r['ise_9'])} \\\\"
                )
            rows.append("\\midrule")
    rows.pop()
    body = "\n".join(rows)
    write("exact_nested", body)

    names = MAIN + RULE
    cols = [f"{m}/{c}" for m in ("heston", "vg", "bates") for c in ("dense", "sparse")]
    lines = []
    for name in names:
        vals = [ex[c].get(name, {}).get("ise") for c in cols]
        lines.append(LABEL[name] + " & " + " & ".join(_bold_min(vals, _sci) if False else [_sci(v) for v in vals]) + " \\\\")
    # bold the column minimum across methods
    mat = np.array([[ex[c].get(n, {}).get("ise", np.nan) for c in cols] for n in names], float)
    out = []
    for i, name in enumerate(names):
        cells = []
        for j, c in enumerate(cols):
            v = mat[i, j]
            s = _sci(v)
            if np.isfinite(v) and v <= np.nanmin(mat[:, j]) * (1 + 1e-12):
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        out.append(LABEL[name] + " & " + " & ".join(cells) + " \\\\")
        if name == MAIN[-1]:
            out.append("\\midrule")
    write("exact_comp", "\n".join(out))


# ---------------------------------------------------------------- Monte Carlo


def _mc_cell(reps, name, scale=1e4):
    v = np.array([r[name]["ise"] for r in reps if name in r], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean() * scale), float(v.std(ddof=1) / np.sqrt(v.size) * scale), int(v.size)


def mc_tables():
    mc = _load("mc")
    if mc is None:
        return
    from joe import TRANSFER

    def one(tags, fname, with_reps=True):
        present = set()
        for tag in tags:
            if tag in mc:
                for r in mc[tag]["reps"]:
                    present.update(r)
        names = [n for n in MAIN + RULE + TRANSFER if n in present]
        mat = np.full((len(names), len(tags)), np.nan)
        se = np.full_like(mat, np.nan)
        nrep = {}
        for j, tag in enumerate(tags):
            if tag not in mc:
                continue
            for i, name in enumerate(names):
                m, s, k = _mc_cell(mc[tag]["reps"], name)
                mat[i, j], se[i, j] = m, s
                nrep[tag] = max(nrep.get(tag, 0), k)
        out = []
        for i, name in enumerate(names):
            cells = []
            for j in range(len(tags)):
                v = mat[i, j]
                if not np.isfinite(v):
                    cells.append("{--}")
                    continue
                s = f"{v:.2f}" if v < 100 else f"{v:.0f}"
                col = mat[:, j]
                # bold every entry within two standard errors of the column minimum
                jmin = int(np.nanargmin(col))
                if v - mat[jmin, j] <= 2.0 * se[jmin, j] + 1e-12:
                    s = f"\\textbf{{{s}}}"
                cells.append(f"{s} ({se[i, j]:.2f})" if se[i, j] < 10 else f"{s} ({se[i, j]:.0f})")
            out.append(LABEL[name] + " & " + " & ".join(cells) + " \\\\")
            if name == MAIN[-1] and len(names) > len(MAIN):
                out.append("\\midrule")
        write(fname, "\n".join(out))
        return nrep

    tags = [f"{m}/{c}/0.01" for m in ("heston", "vg", "bates") for c in ("dense", "sparse", "listed")]
    n1 = one(tags, "mc_main")
    tags = [f"{m}/{c}/0.005" for m in ("heston", "vg", "bates") for c in ("dense", "sparse")]
    n2 = one(tags, "mc_noise_a")
    tags = [f"{m}/{c}/0.02" for m in ("heston", "vg", "bates") for c in ("dense", "sparse")]
    n2.update(one(tags, "mc_noise_b"))
    (TAB / "mc_reps.tex").write_text(
        f"\\newcommand{{\\mcrepsmain}}{{{min(n1.values()) if n1 else 0}}}"
        f"\\newcommand{{\\mcrepsnoise}}{{{min(n2.values()) if n2 else 0}}}\n"
    )
    # negative mass and penalty frequency of ours at sd = 0.01
    lines = []
    for m in ("heston", "vg", "bates"):
        for c in ("dense", "sparse", "listed"):
            tag = f"{m}/{c}/0.01"
            if tag not in mc:
                continue
            reps = mc[tag]["reps"]
            neg = np.array([r["ours_rule"].get("neg", np.nan) for r in reps], float)
            cpos = np.array([r["ours_rule"].get("c", np.nan) for r in reps], float)
            h = np.array([r["ours_rule"]["h"] for r in reps], float)
            lines.append(
                f"{MODEL[m]}, {c} & {np.nanmean(h):.2f} & {100*np.nanmean(cpos > 0):.0f} & "
                f"{np.nanmean(neg):.4f} & {np.nanmax(neg):.4f} \\\\"
            )
    write("mc_ours_diag", "\n".join(lines))


def oracle_table():
    mc = _load("mc")
    orc = _load("mc_oracle")
    if mc is None or orc is None:
        return
    keys = ["ours", "pca", "asl", "asd", "ghs_iv", "ghs_call"]
    names = {"ours": "Ours", "pca": "PCA", "asl": "AS--Lo", "asd": "AS--Duarte", "ghs_iv": "GHS IV", "ghs_call": "GHS call"}
    feas = {"ours": "ours_rule", "pca": "pca_cv", "asl": "asl_cv", "asd": "asd_cv", "ghs_iv": "ghs_iv_cv", "ghs_call": "ghs_call_cv"}
    tags = [f"{m}/{c}/0.01" for m in ("heston", "vg", "bates") for c in ("dense", "sparse", "listed")]
    tags = [t for t in tags if t in orc]
    mat = np.full((len(keys), len(tags)), np.nan)
    se = np.full_like(mat, np.nan)
    for j, tag in enumerate(tags):
        for i, k in enumerate(keys):
            v = np.array([r[k]["ise"] for r in orc[tag]["reps"]], float) * 1e4
            v = v[np.isfinite(v)]
            mat[i, j], se[i, j] = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
    out = []
    for i, k in enumerate(keys):
        cells = []
        for j, tag in enumerate(tags):
            v = mat[i, j]
            s = f"{v:.2f}" if v < 100 else f"{v:.0f}"
            jmin = int(np.nanargmin(mat[:, j]))
            if v - mat[jmin, j] <= 2.0 * se[jmin, j] + 1e-12:
                s = f"\\textbf{{{s}}}"
            cells.append(f"{s} ({se[i, j]:.2f})" if se[i, j] < 10 else f"{s} ({se[i, j]:.0f})")
        out.append(names[k] + " & " + " & ".join(cells) + " \\\\")
    write("mc_oracle", "\n".join(out))
    (TAB / "mc_oracle_cols.tex").write_text(" & ".join(f"{MODEL[t.split('/')[0]]} {t.split('/')[1]}" for t in tags) + "\n")


def variants_table():
    mc = _load("variants")
    li = _load("variants_listed")
    if mc is None and li is None:
        return
    lines = []
    if mc is not None:
        for design, cols in (("iv", ["rule", "rule_pos", "gcv", "gcv_pos"]),
                             ("price", ["rule", "rule_pos", "gcv", "rule_w", "rule_w_pos", "gcv_w"])):
            head = {"rule": "rule", "rule_pos": "rule+", "gcv": "GCV", "gcv_pos": "GCV+", "rule_w": "rule, w",
                    "rule_w_pos": "rule, w+", "gcv_w": "GCV, w"}
            lines.append("\\begin{tabular}{@{}l " + "r" * len(cols) + "@{}}")
            lines.append("\\toprule")
            title = "Implied-volatility noise" if design == "iv" else "Heteroskedastic price noise"
            lines.append(f"\\multicolumn{{{len(cols)+1}}}{{@{{}}l}}{{{title}}} \\\\")
            lines.append("Design & " + " & ".join(head[c] for c in cols) + " \\\\")
            lines.append("\\midrule")
            for m in ("heston", "vg", "bates"):
                for c in ("dense", "sparse", "listed"):
                    tag = f"{design}/{m}/{c}"
                    if tag not in mc:
                        continue
                    reps = mc[tag]["reps"]
                    cells = []
                    vals = []
                    for col in cols:
                        v = np.array([r[col]["ise"] for r in reps if col in r], float) * 1e4
                        vals.append((v.mean(), v.std(ddof=1) / np.sqrt(v.size)) if v.size else (np.nan, np.nan))
                    best = np.nanmin([x[0] for x in vals])
                    for (mu, se) in vals:
                        t = f"{mu:.2f} ({se:.2f})" if np.isfinite(mu) else "{--}"
                        if np.isfinite(mu) and mu <= best + 1e-12:
                            t = f"\\textbf{{{t}}}"
                        cells.append(t)
                    lines.append(f"{MODEL[m]}, {c} & " + " & ".join(cells) + " \\\\")
            lines.append("\\bottomrule")
            lines.append("\\end{tabular}\\par\\medskip")
    if li is not None:
        cols = ["rule", "gcv", "rule_w", "gcv_w"]
        lines.append("\\begin{tabular}{@{}l rrrr rrrr@{}}")
        lines.append("\\toprule")
        lines.append("\\multicolumn{9}{@{}l}{Listed chains} \\\\")
        lines.append(" & \\multicolumn{4}{c}{Hold-out, half-spreads} & \\multicolumn{4}{c}{Modes} \\\\")
        lines.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}")
        lines.append("Chain & rule & GCV & rule, w & GCV, w & rule & GCV & rule, w & GCV, w \\\\")
        lines.append("\\midrule")
        for title, rows in li.items():
            hs = [rows[c]["ho_spread"] for c in cols]
            pk = [rows[c]["peaks"] for c in cols]
            lines.append(title + " & " + " & ".join(f"{v:.3f}" for v in hs) + " & " + " & ".join(str(v) for v in pk) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
    write("variants", "\n".join(lines))


# ---------------------------------------------------------------- listed


def _listed_rows(data, names, chain_titles, with_pts=True):
    out = []
    for title in chain_titles:
        rows = data[title]["rows"]
        hs = [rows[n]["ho_spread"] for n in names]
        pts = [rows[n]["ho_pts"] for n in names]
        bp = [1e4 * rows[n]["ho_iv"] for n in names]
        ins = [rows[n]["otm_spread"] for n in names]
        hs_b = _bold_min(hs, lambda v: _f(v, 2))
        pts_b = _bold_min(pts, lambda v: _f(v, 3))
        bp_b = _bold_min(bp, lambda v: _f(v, 0))
        for i, n in enumerate(names):
            r = rows[n]
            first = title if i == 0 else ""
            cells = [first, LABEL[n]]
            if with_pts:
                cells.append(pts_b[i])
            cells += [hs_b[i], bp_b[i], _f(ins[i], 2), _f(r["mass"], 2), _f(r["neg"], 3), str(r["peaks"]), _f(r["tv"], 2)]
            out.append(" & ".join(cells) + " \\\\")
        out.append("\\midrule")
    out.pop()
    return out


def listed_tables():
    data = _load("listed")
    if data is None:
        return
    titles = list(data)
    write("listed_main", "\n".join(_listed_rows(data, MAIN, titles)))
    write("listed_rule", "\n".join(_listed_rows(data, RULE, titles)))
    tr = _load("listed_transfer")
    if tr is not None:
        from joe import TRANSFER
        write("listed_transfer", "\n".join(_listed_rows(tr, TRANSFER, list(tr))))
    # bandwidths of every method, in strike units, and ours penalty
    lines = []
    for title in titles:
        rows = data[title]["rows"]
        m = data[title]["meta"]
        cells = [title, str(m["n"]), f"{m['F']:.0f}", f"{m['T']:.3f}", f"{100*m['sigma']:.1f}",
                 f"{m['median_half_spread']:.2f}"]
        cells += [_f(rows[n]["h"], 1) for n in MAIN]
        cells.append(_sci(rows["ours_rule"]["c"]))
        lines.append(" & ".join(cells) + " \\\\")
    write("listed_meta", "\n".join(lines))


# ---------------------------------------------------------------- extra chains


def extra_tables():
    data = _load("extra")
    if data is None:
        return
    titles = list(data)
    names = [n for n in MAIN if all(n in data[t]["rows"] for t in titles)]
    hs = np.array([[data[t]["rows"][n]["ho_spread"] for n in names] for t in titles], float)
    bp = np.array([[1e4 * data[t]["rows"][n]["ho_iv"] for n in names] for t in titles], float)
    neg = np.array([[data[t]["rows"][n]["neg"] for n in names] for t in titles], float)
    uni = np.array([[data[t]["rows"][n]["peaks"] == 1 for n in names] for t in titles], float)
    ranks = np.argsort(np.argsort(hs, axis=1), axis=1) + 1
    best = (hs <= hs.min(axis=1, keepdims=True) * (1 + 1e-12)).sum(axis=0)
    lines = []
    for j, n in enumerate(names):
        lines.append(
            f"{LABEL[n]} & {np.mean(hs[:, j]):.2f} & {np.median(hs[:, j]):.2f} & {np.mean(ranks[:, j]):.2f} & "
            f"{int(best[j])} & {np.median(bp[:, j]):.0f} & {100*np.mean(uni[:, j]):.0f} & {np.mean(neg[:, j]):.3f} \\\\"
        )
    write("extra_summary", "\n".join(lines))
    (TAB / "extra_count.tex").write_text(f"\\newcommand{{\\nextra}}{{{len(titles)}}}\n")
    lines = []
    for i, t in enumerate(titles):
        m = data[t]["meta"]
        cells = [t.replace(" ", "~", 1), str(m["n"]), f"{m['T']:.2f}"]
        cells += _bold_min(list(hs[i]), lambda v: _f(v, 2))
        lines.append(" & ".join(cells) + " \\\\")
    write("extra_full", "\n".join(lines))
    (TAB / "extra_header.tex").write_text(" & ".join(SHORT.get(n, LABEL[n]) for n in names) + "\n")


# ---------------------------------------------------------------- sensitivity


def sens_tables():
    data = _load("sensitivity")
    if data is None:
        return
    consts = sorted({float(k.split("/")[0]) for t in data["listed"] for k in data["listed"][t]})
    lines = []
    for title in data["listed"]:
        for pen in ("rule", "zero"):
            cells = [title if pen == "rule" else "", "rule" if pen == "rule" else "zero"]
            for c in consts:
                r = data["listed"][title][f"{c}/{pen}"]
                cells.append(f"{r['ho_spread']:.3f}")
            for c in consts:
                r = data["listed"][title][f"{c}/{pen}"]
                cells.append(f"{r['peaks']}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines.pop()
    write("sens_listed", "\n".join(lines))
    lines = []
    for tag in data["mc"]:
        for pen in ("rule", "zero"):
            model, chain = tag.split("/")
            cells = [f"{MODEL[model]}, {chain}" if pen == "rule" else "", "rule" if pen == "rule" else "zero"]
            for c in consts:
                r = data["mc"][tag][f"{c}/{pen}"]
                cells.append(f"{1e4*r['mean']:.2f}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines.pop()
    write("sens_mc", "\n".join(lines))
    (TAB / "sens_consts.tex").write_text(" & ".join(f"{c:.2f}" for c in consts) + "\n")


if __name__ == "__main__":
    exact_tables()
    mc_tables()
    oracle_table()
    variants_table()
    listed_tables()
    extra_tables()
    sens_tables()
