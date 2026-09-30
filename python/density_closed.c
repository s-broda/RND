#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Gaussian convolution of a flat-extended cubic spline.
   P'' is piecewise linear between the knots, with a slope jump where the
   flat tails meet the spline. Cells are gathered into boxes one narrow
   bandwidth wide. On each box the kernel is the order-8 Hermite expansion
   of the Gaussian, and thricing reuses those moments at the wider widths.
   g2 is the second derivative of the convolved extension (the call density
   is -F g2). level is the convolved extension minus the left endpoint.
   g2_plain is the untwiced second derivative. Density and level may be
   read on different abscissae. Returns 0, or -1 if the workspace cannot
   be allocated. */

static const double INV_SQRT_2PI = 0.3989422804014327;
enum { ORDER = 8, NMOM = ORDER + 1 };
#define WINDOW 5.0

/* Cephes exponential. The kernel only evaluates exp(-u^2/2) for |u| of a few
   units, where this rational form is within one ulp of libm and several times
   faster. */
static double fast_exp(double x) {
    const double log2e = 1.4426950408889634073599;
    const double c1 = 6.93145751953125e-1;
    const double c2 = 1.42860682030941723212e-6;
    double n, r, xx, px, qx, p;
    int e;
    union { double d; uint64_t u; } v;
    if (x < -40.0) return 0.0;
    n = floor(log2e * x + 0.5);
    r = x - n * c1 - n * c2;
    xx = r * r;
    px = 1.26177193074810590878e-4;
    px = px * xx + 3.02994407707441961300e-2;
    px = px * xx + 9.99999999999999999910e-1;
    qx = 3.00198505138664455042e-6;
    qx = qx * xx + 2.52448340349684104192e-3;
    qx = qx * xx + 2.27265548208155028766e-1;
    qx = qx * xx + 2.00000000000000000009e0;
    p = px * r;
    r = 1.0 + 2.0 * p / (qx - p);
    e = (int)n + 1023;
    if (e <= 0) return 0.0;
    v.u = (uint64_t)e << 52;
    return r * v.d;
}

static double ndtr(double u) {
    if (u > 8.0) return 1.0;
    if (u < -8.0) return 0.0;
    return 0.5 * (1.0 + erf(u * 0.7071067811865476));
}

static inline __attribute__((always_inline)) void accum_box(
    const double *row, const double *nrow,
    double u, double phi, double invh, double *acc, double *lacc)
{
    double tau_nm2 = phi;
    double tau = u * phi;
    double hpow = invh;
    int k;
    *acc += row[0] * tau_nm2 * hpow;
    if (nrow) *lacc += nrow[0] * tau_nm2 * hpow;
    hpow *= invh;
    *acc += row[1] * tau * hpow;
    if (nrow) *lacc += nrow[1] * tau * hpow;
    for (k = 1; k < ORDER; k++) {
        double next = (u * tau - tau_nm2) / (double)(k + 1);
        tau_nm2 = tau;
        tau = next;
        hpow *= invh;
        *acc += row[k + 1] * tau * hpow;
        if (nrow) *lacc += nrow[k + 1] * tau * hpow;
    }
}

static void zero_out(double *g2, int n, double *level, int nl, double *g2_plain) {
    int i;
    if (g2) {
        for (i = 0; i < n; i++) g2[i] = 0.0;
    }
    if (level) {
        for (i = 0; i < nl; i++) level[i] = 0.0;
    }
    if (g2_plain) {
        for (i = 0; i < n; i++) g2_plain[i] = 0.0;
    }
}

static inline __attribute__((always_inline)) void one_bandwidth(
    double xi, double hh, double invh, int W, double origin, double width,
    const double *mu, const double *nu, int nbox,
    double t0, double tm, double p1L, double p1R,
    double left, double right, int want_level,
    double *acc_out, double *lacc_out)
{
    double acc = 0.0, lacc = 0.0, u;
    int b = (int)llround((xi - origin) / width);
    int off;
    for (off = -W; off <= W; off++) {
        int bb = b + off;
        double center, phi;
        if ((unsigned)bb >= (unsigned)nbox) continue;
        center = origin + (double)bb * width;
        u = (xi - center) / hh;
        phi = fast_exp(-0.5 * u * u) * INV_SQRT_2PI;
        accum_box(
            mu + (size_t)bb * NMOM,
            (want_level && nu) ? nu + (size_t)bb * NMOM : NULL,
            u, phi, invh, &acc, &lacc);
    }
    u = (xi - t0) / hh;
    acc += p1L * fast_exp(-0.5 * u * u) * INV_SQRT_2PI * invh;
    u = (xi - tm) / hh;
    acc -= p1R * fast_exp(-0.5 * u * u) * INV_SQRT_2PI * invh;
    if (want_level) {
        double u0 = (xi - t0) / hh;
        double u1 = (xi - tm) / hh;
        lacc += -left * ndtr(u0) + right * ndtr(u1);
    }
    *acc_out = acc;
    *lacc_out = lacc;
}

static void eval_points(
    const double *x, int n,
    const double *t, const double *p1, int m,
    double h, const double *fac, const double *wt, int nfac,
    const double *mu, const double *nu, int nbox, double width,
    double left, double right, int want_level, int want_plain,
    double *g2, double *level, double *g2_plain)
{
    /* Two strikes at a time. Each Hermite chain is serial, and a performance
       core only stays full when those chains are independent. */
    double origin = t[0] + 0.5 * width;
    double t0 = t[0], tm = t[m - 1], p1L = p1[0], p1R = p1[m - 1];
    int i = 0, f;
    for (; i + 1 < n; i += 2) {
        double sum0 = 0.0, sum1 = 0.0, lvl0 = 0.0, lvl1 = 0.0;
        double plain0 = 0.0, plain1 = 0.0;
        int plain_set = 0;
        double x0 = x[i], x1 = x[i + 1];
        for (f = 0; f < nfac; f++) {
            double hh = h * fac[f];
            double invh = 1.0 / hh;
            double acc0, lacc0, acc1, lacc1;
            int W = (int)ceil(WINDOW * hh / width);
            one_bandwidth(
                x0, hh, invh, W, origin, width, mu, nu, nbox,
                t0, tm, p1L, p1R, left, right, want_level, &acc0, &lacc0);
            one_bandwidth(
                x1, hh, invh, W, origin, width, mu, nu, nbox,
                t0, tm, p1L, p1R, left, right, want_level, &acc1, &lacc1);
            sum0 += wt[f] * acc0;
            sum1 += wt[f] * acc1;
            if (want_level) {
                lvl0 += wt[f] * lacc0;
                lvl1 += wt[f] * lacc1;
            }
            if (want_plain && !plain_set && fac[f] == 1.0) {
                plain0 = acc0;
                plain1 = acc1;
                plain_set = 1;
            }
        }
        if (g2) {
            g2[i] = sum0;
            g2[i + 1] = sum1;
        }
        if (level) {
            level[i] = lvl0;
            level[i + 1] = lvl1;
        }
        if (g2_plain) {
            g2_plain[i] = plain0;
            g2_plain[i + 1] = plain1;
        }
    }
    for (; i < n; i++) {
        double sum = 0.0, lvl = 0.0, plain = 0.0;
        int plain_set = 0;
        for (f = 0; f < nfac; f++) {
            double hh = h * fac[f];
            double invh = 1.0 / hh;
            double acc, lacc;
            int W = (int)ceil(WINDOW * hh / width);
            one_bandwidth(
                x[i], hh, invh, W, origin, width, mu, nu, nbox,
                t0, tm, p1L, p1R, left, right, want_level, &acc, &lacc);
            sum += wt[f] * acc;
            if (want_level) lvl += wt[f] * lacc;
            if (want_plain && !plain_set && fac[f] == 1.0) {
                plain = acc;
                plain_set = 1;
            }
        }
        if (g2) g2[i] = sum;
        if (level) level[i] = lvl;
        if (g2_plain) g2_plain[i] = plain;
    }
}

/* Power basis of a cubic piece: c[0] is the cubic coefficient, c[3] the value.
   Layout matches a C-contiguous SciPy PPoly coefficient array of shape (4, m-1). */
void ppoly_jets(
    const double *c, const double *x, int m,
    double *p0, double *p1, double *sec)
{
    int i, nm;
    double dt, c0, c1, c2, c3;
    if (m < 2) return;
    nm = m - 1;
    for (i = 0; i < nm; i++) {
        p0[i] = c[3 * nm + i];
        p1[i] = c[2 * nm + i];
        sec[i] = 2.0 * c[nm + i];
    }
    dt = x[m - 1] - x[m - 2];
    c0 = c[nm - 1];
    c1 = c[nm + nm - 1];
    c2 = c[2 * nm + nm - 1];
    c3 = c[3 * nm + nm - 1];
    p0[m - 1] = ((c0 * dt + c1) * dt + c2) * dt + c3;
    p1[m - 1] = (3.0 * c0 * dt + 2.0 * c1) * dt + c2;
    sec[m - 1] = 6.0 * c0 * dt + 2.0 * c1;
}

static double de_boor(const double *t, int nt, const double *c, int nc, int k, double x) {
    int lo = 0, hi = nt, i, j, r;
    double d[8];
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (t[mid] <= x) lo = mid + 1;
        else hi = mid;
    }
    i = lo - 1;
    if (i < k) i = k;
    if (i > nc - 1) i = nc - 1;
    for (j = 0; j <= k; j++) d[j] = c[i - k + j];
    for (r = 1; r <= k; r++) {
        for (j = k; j >= r; j--) {
            double left = t[i - k + j];
            double right = t[i + 1 + j - r];
            double den = right - left;
            double alpha = den == 0.0 ? 0.0 : (x - left) / den;
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }
    return d[k];
}

static void deriv_coef(const double *t, const double *c, int nc, int k, double *cp) {
    int j;
    for (j = 0; j < nc - 1; j++) {
        double den = t[j + k + 1] - t[j + 1];
        cp[j] = den == 0.0 ? 0.0 : (double)k * (c[j + 1] - c[j]) / den;
    }
}

/* Cubic B-spline jets at x. t has length nc+k+1. Returns 0, or -1. */
int bspline_jets(
    const double *t, int nt, const double *c, int nc, int k,
    const double *x, int m,
    double *p0, double *p1, double *sec)
{
    double *c1, *c2;
    int i, nc1, nc2, nt1, nt2;
    if (k != 3 || nc < 4 || nt != nc + k + 1 || m < 1) return -1;
    c1 = (double *)malloc((size_t)(nc - 1) * sizeof(double));
    c2 = (double *)malloc((size_t)(nc - 2) * sizeof(double));
    if (!c1 || !c2) {
        free(c1);
        free(c2);
        return -1;
    }
    deriv_coef(t, c, nc, 3, c1);
    nc1 = nc - 1;
    nt1 = nt - 2;
    deriv_coef(t + 1, c1, nc1, 2, c2);
    nc2 = nc - 2;
    nt2 = nt - 4;
    for (i = 0; i < m; i++) {
        p0[i] = de_boor(t, nt, c, nc, 3, x[i]);
        p1[i] = de_boor(t + 1, nt1, c1, nc1, 2, x[i]);
        sec[i] = de_boor(t + 2, nt2, c2, nc2, 1, x[i]);
    }
    free(c1);
    free(c2);
    return 0;
}

int rnd_conv(
    const double *x, int n,
    const double *xl, int nl,
    const double *t, const double *p0, const double *p1, const double *s, int m,
    double h,
    const double *fac, const double *wt, int nfac,
    double *g2, double *level, double *g2_plain)
{
    int nbox, b, j, k, f, bL, bR;
    double width, *mu, *nu;
    double L, R, slope, A, a, ccut, eL, base, Bb, zL, zR, pwL, pwR;
    double shift, a0, a1, a2, a3, c0, c1, c2, c3, left, right;
    int want_level = (level && nl > 0 && xl);

    if (m < 2 || nfac < 1 || !(h > 1e-12) || !(t[m - 1] > t[0])) {
        zero_out(g2, n, level, nl, g2_plain);
        return 0;
    }
    for (f = 0; f < nfac; f++) {
        if (!(fac[f] > 1e-12)) {
            zero_out(g2, n, level, nl, g2_plain);
            return 0;
        }
    }
    nbox = (int)ceil((t[m - 1] - t[0]) / h);
    if (nbox < 1) nbox = 1;
    width = (t[m - 1] - t[0]) / (double)nbox;
    mu = (double *)calloc((size_t)nbox * NMOM, sizeof(double));
    nu = want_level ? (double *)calloc((size_t)nbox * NMOM, sizeof(double)) : NULL;
    if (!mu || (want_level && !nu)) {
        free(mu);
        free(nu);
        return -1;
    }
    for (j = 0; j < m - 1; j++) {
        L = t[j];
        R = t[j + 1];
        if (!(R > L)) continue;
        slope = (s[j + 1] - s[j]) / (R - L);
        A = s[j] - slope * L;
        a0 = p0[j];
        a1 = p1[j];
        a2 = 0.5 * s[j];
        a3 = slope / 6.0;
        bL = (int)floor((L - t[0]) / width);
        bR = (int)floor(((R - t[0]) / width) - 1e-12);
        if (bL < 0) bL = 0;
        if (bR >= nbox) bR = nbox - 1;
        for (b = bL; b <= bR; b++) {
            double coef[4];
            eL = t[0] + (double)b * width;
            a = L > eL ? L : eL;
            ccut = R < eL + width ? R : eL + width;
            if (!(ccut > a)) continue;
            shift = (eL + 0.5 * width) - L;
            base = A + slope * (eL + 0.5 * width);
            Bb = slope;
            zL = a - (eL + 0.5 * width);
            zR = ccut - (eL + 0.5 * width);
            c0 = ((a3 * shift + a2) * shift + a1) * shift + a0;
            c1 = (3.0 * a3 * shift + 2.0 * a2) * shift + a1;
            c2 = 3.0 * a3 * shift + a2;
            c3 = a3;
            coef[0] = c0;
            coef[1] = c1;
            coef[2] = c2;
            coef[3] = c3;
            pwL = 1.0;
            pwR = 1.0;
            for (k = 0; k <= ORDER; k++) {
                double Ik = (pwR * zR - pwL * zL) / (double)(k + 1);
                double Ik1 = (pwR * zR * zR - pwL * zL * zL) / (double)(k + 2);
                mu[(size_t)b * NMOM + k] += base * Ik + Bb * Ik1;
                if (nu) {
                    double pLw = pwL, pRw = pwR, moment = 0.0;
                    int mm;
                    for (mm = 0; mm < 4; mm++) {
                        pLw *= zL;
                        pRw *= zR;
                        moment += coef[mm] * (pRw - pLw) / (double)(k + mm + 1);
                    }
                    nu[(size_t)b * NMOM + k] += moment;
                }
                pwL *= zL;
                pwR *= zR;
            }
        }
    }
    left = p0[0];
    right = p0[m - 1];
    if (want_level && xl == x && nl == n) {
        eval_points(
            x, n, t, p1, m, h, fac, wt, nfac, mu, nu, nbox, width,
            left, right, 1, g2_plain != NULL, g2, level, g2_plain);
    } else {
        if ((g2 || g2_plain) && n > 0) {
            eval_points(
                x, n, t, p1, m, h, fac, wt, nfac, mu, NULL, nbox, width,
                left, right, 0, g2_plain != NULL, g2, NULL, g2_plain);
        }
        if (want_level) {
            eval_points(
                xl, nl, t, p1, m, h, fac, wt, nfac, mu, nu, nbox, width,
                left, right, 1, 0, NULL, level, NULL);
        }
    }
    free(mu);
    free(nu);
    return 0;
}
