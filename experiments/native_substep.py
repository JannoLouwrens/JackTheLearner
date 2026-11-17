"""native_substep.py — J6 item (2b), piece 1 of 4: the per-sub-step contact
scans in C, read straight from MuJoCo's own `mjData*` (ORDERS J6).

`experiments/substep_split.py` priced (2b): the native CALL saves nothing;
every gain is in four per-sub-step Python pieces (`drives.substep`,
`water.apply`, `synth.step`, `W0._grounded`). Identity forbids moving any of
them to once a decision, so they move into C one at a time, each proven
identical against today's Python on live trajectories BEFORE any `jack_env`
option wires them in (a `jack_env` edit stales its certificates — one edit,
when all four are proven, not four).

This file is that pipeline plus its first functions (piece 1 integer-exact):
  jn_grounded(d, body_mask, ground_mask)  == W0._grounded()
  jn_partners(d, mask, out)               == drives._contact_partners(d, mask)
                                             (as a per-geom 0/1 row)
C reads `d->contact[k].geom[0..1]` through `MjData._address` — no copy, no
numpy view. Built with gcc + cffi ABI mode (`dlopen`) against the venv's own mujoco 3.2.3
`include/` and `libmujoco.so.3.2.3`, output under /data/caches/jack_native
(never /home or /tmp: `/` is near its floor).

Not world code; nothing here is imported by the world. Proof:
`tests/test_native_substep.py` (identity on live JackEnv sub-steps, 3 seeds,
plus a control that must differ).

Float order for piece 2 (`drives.substep`), MEASURED 11-17 09:42 (probe on
live sub-steps, seeds 153-155, 12,000 calls; and 20,000 random vectors):
  power = np.abs(tau*omega).sum(): tau/omega are ndof-6 = 4 long, under
    numpy's pairwise block (8), so it is a plain left-to-right loop —
    C sequential sum matches 12,000/12,000.
  np.linalg.norm(qvel) (10 long) and norm(qvel[:3]): NO plain ordering
    matches (sequential 1,776 and 1,076 misses of 12,000; 2/4/8-accumulator
    trees worse) — this box's OpenBLAS ddot accumulates with FMA. C
    `s = fma(x[i], x[i], s)` left to right, then sqrt, compiled with
    `-ffp-contract=off`, matches 20,000/20,000 (n = 10 and 3).
  This is the box's BLAS. Kaggle's x86 BLAS may order differently: the
  identity test must run where the option runs, or the option stays box-only.

Measured 11-17 09:38, box, nice 19 (seed 153, ncon 23): `_grounded` Python
13.69 us, C 1.02 us a call -> x40 = 0.51 ms of a decision (of 0.73 priced).
Identity (`tests/test_native_substep.py`): 18,000/18,000 sub-step calls of
each function match on seeds 153-155; the control (ground mask cleared)
differs on all 17,456 grounded sub-steps.

Piece 2, MEASURED 11-17 10:40: `jn_drives_substep` == `drives.DriveLayer.
substep` (power, rest, impact edge, eating over the partners row + food
timers, submersion clock/drowning, wetness), state in a C struct
(`NativeDrives`: load from the Python layer, diff field by field).
`tests/test_native_drives.py` PASS: 54,000/54,000 sub-steps identical with
exact float equality on every field (C loaded once a decision, then on its
own for 40 sub-steps): seeds 153-155 x {today's world random actions; body
set on a live floor food every 10 decisions (2 meals a seed); pool with the
drive layer's surface raised 0 -> 0.5 m in BOTH (W0's floor plane covers
the pool, so in today's world no geom is ever wet — the branch is planted)}.
Control (plain-order norm) differs on 370-775 sub-steps a seed.
Speed (box, nice 19, seeds 153-155, 20,000 calls): Python 44.4 us, C 1.17 us
a call -> 1.73 ms of a decision.

Piece 3, MEASURED 11-17 10:45: `jn_water_apply` == `playground.Water.apply`
(per-body pool test, lazy held-dry contact scan, spherical-cap buoyancy,
quadratic + linear drag, the W0.PL additive write into xfrc_applied; `r**3`
is C pow(), speed is the FMA-order norm). `NativeWater` loads `_last`,
`_written`, `_last_any` ONCE per env and evolves alone for the whole run.
`tests/test_native_water.py` PASS: 54,000/54,000 sub-steps identical (whole
xfrc_applied + state, exact): seeds 153-155 x {random actions; dropped in
the pool at surface 0 — held dry on 5,571/6,000 calls; surface raised to
0.5 m with the dry set recomputed — force on 6,000/6,000}. Control (plain
norm) differs on 5,487-5,492 a seed (float case). FINDING: in today's
world the water force DOES fire — seeds 154/155 put force on 68/63
sub-steps of 6,000 (a limb whose bounding sphere dips below z=0 over the
pool while not touching the floor), though `drives` never sees him wet.
Speed (box, nice 19, 20,000 calls): Python 14.7-18.5 us, C 0.90 us a call
-> 0.55-0.70 ms of a decision.

Piece 4, MEASURED 11-17 10:27: `jn_synth_step` == `fast_sim._lazy_step` (the
shipped option under ruling 34). C owns the pair grouping (dict insertion
order) and `_prev_pairs`/`_last_fired`; for each FIRING pair Python builds
the event with the shipped expressions (events are rare: 0.5-7% of
sub-steps). `tests/test_native_synth.py` PASS: 36,000/36,000 sub-steps,
every event field + state exact, C loaded once per synth: seeds 153-155 x
{random: 322-440 events; set on food every 10 decisions: 30-47 events}.
Control (refractory window 0) fires 4-828 extra events a seed. NOT
exercised: a reset inside a run (no life ended in 150 decisions) — the
wiring's whole-trajectory test must cover it.
Speed (box, nice 19, 20,000 calls): Python 8.1-11.3 us, C 1.31-1.34 us a
call -> 0.27-0.40 ms of a decision.

Usage: $PY -m experiments.native_substep   # build + timing on one live env
"""
from __future__ import annotations

import os
import sys
import time

import mujoco
import numpy as np

BUILD_DIR = os.environ.get("JACK_NATIVE_DIR", "/data/caches/jack_native")  # Kaggle: a writable dir
_MJ_DIR = os.path.dirname(mujoco.__file__)

CDEF = """
int jn_grounded(uintptr_t d, const uint8_t* body, const uint8_t* ground);
int jn_partners(uintptr_t d, const uint8_t* mask, uint8_t* out, int ngeom);
typedef struct {
    int dofadr, ndof, ngeom, fma_norm;
    const uint8_t* jack_mask; const uint8_t* jack_geoms;
    int nfood; const int* food_gid; const double* food_nu;
    const double* food_respawn; double* respawn_at; int* ate_total;
    int has_pool; double px, py, half, surf;
    int nhead; const int* head; int njack; const int* jack_gids;
    double q_rest, drown_delay, tau_wet, tau_dry;
    uint8_t* partners;
} jn_dcfg;
typedef struct {
    double power_dt, dt_acc, rest_dt, j_max, ate, drown_dt, w, t;
    double prev_speed, submerged_since;
    int n_onsets, touching_world, has_prev_speed, has_submerged, wet_in;
} jn_dstate;
void jn_drives_substep(uintptr_t d, const jn_dcfg* c, jn_dstate* s, double dt);
typedef struct {
    int nbody, ngeom, fma_norm, enabled;
    const double* radius; const double* mass; const uint8_t* dry;
    const int* geom_body;
    double x, y, half, surf, cd, c_lin, rho, g;
    double* last; double* written; int last_any;
    double* newf; uint8_t* held;
} jn_water;
void jn_water_apply(uintptr_t d, jn_water* w);
typedef struct {
    int ngeom, nprev, ncur;
    uint8_t* prev_mark; double* last; int* prev; int* cur; int* seen;
    int stamp; double refr;
} jn_synth;
int jn_synth_step(uintptr_t d, jn_synth* s, int* fire, int maxfire);
"""

SRC = r"""
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mujoco/mujoco.h>

int jn_grounded(uintptr_t dp, const uint8_t* body, const uint8_t* ground) {
    const mjData* d = (const mjData*)dp;
    for (int k = 0; k < d->ncon; k++) {
        int a = d->contact[k].geom[0], b = d->contact[k].geom[1];
        if ((body[a] && ground[b]) || (body[b] && ground[a])) return 1;
    }
    return 0;
}

int jn_partners(uintptr_t dp, const uint8_t* mask, uint8_t* out, int ngeom) {
    const mjData* d = (const mjData*)dp;
    int n = 0;
    memset(out, 0, (size_t)ngeom);
    for (int k = 0; k < d->ncon; k++) {
        int a = d->contact[k].geom[0], b = d->contact[k].geom[1];
        if (mask[a] && !out[b]) { out[b] = 1; n++; }
        if (mask[b] && !out[a]) { out[a] = 1; n++; }
    }
    return n;
}

typedef struct {
    int dofadr, ndof, ngeom, fma_norm;
    const uint8_t* jack_mask; const uint8_t* jack_geoms;
    int nfood; const int* food_gid; const double* food_nu;
    const double* food_respawn; double* respawn_at; int* ate_total;
    int has_pool; double px, py, half, surf;
    int nhead; const int* head; int njack; const int* jack_gids;
    double q_rest, drown_delay, tau_wet, tau_dry;
    uint8_t* partners;
} jn_dcfg;
typedef struct {
    double power_dt, dt_acc, rest_dt, j_max, ate, drown_dt, w, t;
    double prev_speed, submerged_since;
    int n_onsets, touching_world, has_prev_speed, has_submerged, wet_in;
} jn_dstate;

/* np.linalg.norm on this box = OpenBLAS ddot: sequential FMA, then sqrt.
   fma_norm = 0 is the plain order, kept ONLY as the identity test's control. */
static double jn_norm(const double* x, int n, int fma_norm) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s = fma_norm ? fma(x[i], x[i], s) : s + x[i] * x[i];
    return sqrt(s);
}

static int jn_in_pool(const jn_dcfg* c, const double* p) {
    return fabs(p[0] - c->px) <= c->half && fabs(p[1] - c->py) <= c->half
           && p[2] < c->surf;
}

/* drives.DriveLayer.substep, line for line, same float order. */
void jn_drives_substep(uintptr_t dp, const jn_dcfg* c, jn_dstate* s, double dt) {
    const mjData* d = (const mjData*)dp;
    const double* qv = d->qvel + c->dofadr;
    const double* qf = d->qfrc_actuator + c->dofadr;
    double power = 0.0;
    for (int i = 6; i < c->ndof; i++) power += fabs(qf[i] * qv[i]);
    s->power_dt += power * dt;
    s->dt_acc += dt;
    if (jn_norm(qv, c->ndof, c->fma_norm) < c->q_rest) s->rest_dt += dt;

    jn_partners(dp, c->jack_mask, c->partners, c->ngeom);
    double speed = jn_norm(qv, 3, c->fma_norm);
    int touching = 0;
    for (int g = 0; g < c->ngeom; g++)
        if (c->partners[g] && !c->jack_geoms[g]) { touching = 1; break; }
    if (touching && !s->touching_world) {
        double arrival = s->has_prev_speed ? s->prev_speed : speed;
        if (arrival > s->j_max) s->j_max = arrival;
        s->n_onsets += 1;
    }
    s->touching_world = touching;
    s->prev_speed = speed; s->has_prev_speed = 1;

    for (int k = 0; k < c->nfood; k++) {
        if (s->t + s->dt_acc < c->respawn_at[k]) continue;
        if (c->partners[c->food_gid[k]]) {
            s->ate += c->food_nu[k];
            c->respawn_at[k] = s->t + s->dt_acc + c->food_respawn[k];
            c->ate_total[k] += 1;
        }
    }

    int under = 0;
    if (c->has_pool)
        for (int h = 0; h < c->nhead && !under; h++)
            under = jn_in_pool(c, d->geom_xpos + 3 * c->head[h]);
    if (under) {
        double now = s->t + s->dt_acc;
        if (!s->has_submerged) { s->submerged_since = now; s->has_submerged = 1; }
        else if (now - s->submerged_since > c->drown_delay) s->drown_dt += dt;
    } else {
        s->has_submerged = 0;
    }

    int wet = 0;
    if (c->has_pool)
        for (int j = 0; j < c->njack && !wet; j++)
            wet = jn_in_pool(c, d->geom_xpos + 3 * c->jack_gids[j]);
    s->wet_in = wet;
    double w = s->w;
    if (wet) w += (1.0 - w) * dt / c->tau_wet;
    else     w -= w * dt / c->tau_dry;
    s->w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
}

typedef struct {
    int nbody, ngeom, fma_norm, enabled;
    const double* radius; const double* mass; const uint8_t* dry;
    const int* geom_body;
    double x, y, half, surf, cd, c_lin, rho, g;
    double* last; double* written; int last_any;
    double* newf; uint8_t* held;
} jn_water;

/* playground.Water.apply, line for line, same float order.  Python `x ** 3`
   on floats is C pow(); np.linalg.norm(3) is the box BLAS's FMA order. */
void jn_water_apply(uintptr_t dp, jn_water* w) {
    mjData* d = (mjData*)dp;
    if (!w->enabled) return;
    int nb = w->nbody, any_new = 0, held_done = 0;
    const double PI = 3.141592653589793;
    for (int bid = 1; bid < nb; bid++) {
        const double* pos = d->xipos + 3 * bid;
        double r = w->radius[bid];
        if (r <= 0 || w->mass[bid] <= 0) continue;
        if (!(fabs(pos[0] - w->x) < w->half && fabs(pos[1] - w->y) < w->half
              && pos[2] - r < w->surf)) continue;
        if (!held_done) {
            memset(w->held, 0, (size_t)nb);
            for (int k = 0; k < d->ncon; k++) {
                int g1 = d->contact[k].geom[0], g2 = d->contact[k].geom[1];
                if (w->dry[g1]) w->held[w->geom_body[g2]] = 1;
                else if (w->dry[g2]) w->held[w->geom_body[g1]] = 1;
            }
            held_done = 1;
        }
        if (w->held[bid]) continue;
        double dd = w->surf - pos[2], frac;
        if (dd >= r) frac = 1.0;
        else if (dd <= -r) continue;
        else { double h = dd + r; frac = (h * h * (3 * r - h)) / (4.0 * pow(r, 3.0)); }
        double vol = (4.0 / 3.0) * PI * pow(r, 3.0) * frac;
        double buoy = w->rho * vol * w->g;
        const double* vel = d->cvel + 6 * bid + 3;
        double speed = jn_norm(vel, 3, w->fma_norm);
        double area = PI * r * r * frac;
        double k1 = -0.5 * w->rho * w->cd * area * speed;
        double k2 = w->c_lin * w->rho * vol;
        if (!any_new) { memset(w->newf, 0, sizeof(double) * 3 * nb); any_new = 1; }
        for (int i = 0; i < 3; i++) w->newf[3 * bid + i] = k1 * vel[i] - k2 * vel[i];
        w->newf[3 * bid + 2] += buoy;
    }
    if (!any_new) {
        if (!w->last_any) return;
        memset(w->newf, 0, sizeof(double) * 3 * nb);
    }
    int la = 0;
    for (int b = 0; b < nb; b++)
        for (int i = 0; i < 3; i++) {
            double cur = d->xfrc_applied[6 * b + i];
            double base = (cur == w->written[3 * b + i]) ? cur - w->last[3 * b + i] : cur;
            double out = base + w->newf[3 * b + i];
            d->xfrc_applied[6 * b + i] = out;
            w->written[3 * b + i] = out;
            w->last[3 * b + i] = w->newf[3 * b + i];
            if (w->newf[3 * b + i] != 0.0) la = 1;
        }
    w->last_any = la;
}

typedef struct {
    int ngeom, nprev, ncur;
    uint8_t* prev_mark; double* last; int* prev; int* cur; int* seen;
    int stamp; double refr;
} jn_synth;

/* fast_sim._lazy_step's bookkeeping: pairs grouped in order of first
   appearance (== dict insertion order), skipped if touching last sub-step
   or inside the refractory window, `_last_fired`/`_prev_pairs` updated.
   Returns the firing pairs (key a*ngeom+b, a<=b) in order; Python builds
   each event from them with the shipped float ops (events are rare).
   `last` holds -1e9 where `_last_fired` has no key (its .get default). */
int jn_synth_step(uintptr_t dp, jn_synth* s, int* fire, int maxfire) {
    const mjData* d = (const mjData*)dp;
    double t = d->time;
    int ng = s->ngeom, nf = 0;
    s->stamp += 1;
    s->ncur = 0;
    for (int k = 0; k < d->ncon; k++) {
        int a = d->contact[k].geom[0], b = d->contact[k].geom[1];
        int key = a <= b ? a * ng + b : b * ng + a;
        if (s->seen[key] == s->stamp) continue;
        s->seen[key] = s->stamp;
        s->cur[s->ncur++] = key;
    }
    for (int j = 0; j < s->ncur; j++) {
        int key = s->cur[j];
        if (s->prev_mark[key]) continue;
        if (t - s->last[key] < s->refr) continue;
        s->last[key] = t;
        if (nf < maxfire) fire[nf] = key;
        nf++;
    }
    for (int j = 0; j < s->nprev; j++) s->prev_mark[s->prev[j]] = 0;
    for (int j = 0; j < s->ncur; j++) { s->prev_mark[s->cur[j]] = 1; s->prev[j] = s->cur[j]; }
    s->nprev = s->ncur;
    return nf;
}
"""


def build(force: bool = False):
    """Compile SRC once into BUILD_DIR/libjack_native.so; return (ffi, lib).

    ABI mode (gcc + `ffi.dlopen`): the box has no Python headers
    (`pyconfig.h` missing), so cffi's API mode cannot compile here."""
    import subprocess
    from cffi import FFI
    os.makedirs(BUILD_DIR, exist_ok=True)
    so = os.path.join(BUILD_DIR, "libjack_native.so")
    c = os.path.join(BUILD_DIR, "jack_native.c")
    if force or not os.path.exists(so) or open(c).read() != SRC:
        with open(c, "w") as f:
            f.write(SRC)
        subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", so, c,
                        "-I", os.path.join(_MJ_DIR, "include"), "-lm"], check=True)
    ffi = FFI()
    ffi.cdef(CDEF)
    return ffi, ffi.dlopen(so)


class NativeContacts:
    """C twins of `W0._grounded` and `drives._contact_partners` for one env."""

    def __init__(self, body_mask, ground_mask):
        self.ffi, self.lib = build()
        self._body = np.ascontiguousarray(body_mask, dtype=np.uint8)
        self._ground = np.ascontiguousarray(ground_mask, dtype=np.uint8)
        self._pb = self.ffi.from_buffer("uint8_t[]", self._body)
        self._pg = self.ffi.from_buffer("uint8_t[]", self._ground)
        self._out = np.zeros(len(self._body), dtype=np.uint8)
        self._po = self.ffi.from_buffer("uint8_t[]", self._out)

    def grounded(self, data) -> bool:
        return bool(self.lib.jn_grounded(data._address, self._pb, self._pg))

    def partners_row(self, data, mask_u8_ptr) -> np.ndarray:
        self.lib.jn_partners(data._address, mask_u8_ptr, self._po,
                             len(self._out))
        return self._out


class NativeDrives:
    """C twin of `drives.DriveLayer.substep` for ONE layer (piece 2 of 4).

    The layer's per-decision accumulators, its life-level edge/submersion
    state, `t`, `state.w`, and the food timers live in a C struct while the C
    path runs; `load()` copies them in from the Python layer, `diff()` lists
    every field where they disagree (exact float equality). `fma_norm=False`
    is the identity test's control only (plain-order norm, must differ)."""

    _F = ("power_dt", "dt_acc", "rest_dt", "j_max", "ate", "drown_dt")

    def __init__(self, layer, fma_norm: bool = True):
        from experiments import drives
        self.ffi, self.lib = build()
        f, m = self.ffi, layer.model
        self.layer = layer
        self.names = list(layer._food)
        keep = self._keep = []

        def buf(a, ct):
            keep.append(a)
            return f.from_buffer(ct, a)
        jm = np.ascontiguousarray(layer._jack_mask, dtype=np.uint8)
        jg = np.zeros(m.ngeom, dtype=np.uint8)
        jg[list(layer._jack_geoms)] = 1
        self.respawn_at = np.zeros(max(1, len(self.names)))
        self.ate_total = np.zeros(max(1, len(self.names)), dtype=np.int32)
        c = self.cfg = f.new("jn_dcfg*")
        c.dofadr, c.ndof, c.ngeom = layer._dofadr, layer._ndof, m.ngeom
        c.fma_norm = int(bool(fma_norm))
        c.jack_mask, c.jack_geoms = buf(jm, "uint8_t[]"), buf(jg, "uint8_t[]")
        c.nfood = len(self.names)
        c.food_gid = buf(np.array([layer._food[n][0] for n in self.names] or [0],
                                  dtype=np.int32), "int[]")
        c.food_nu = buf(np.array([layer._food[n][1] for n in self.names] or [0.0]),
                        "double[]")
        c.food_respawn = buf(np.array([drives.RESPAWN_S[n] for n in self.names]
                                      or [0.0]), "double[]")
        c.respawn_at = buf(self.respawn_at, "double[]")
        c.ate_total = buf(self.ate_total, "int[]")
        c.has_pool = int(layer.pool is not None)
        if layer.pool is not None:
            c.px, c.py, c.half, c.surf = (float(v) for v in layer.pool)
        c.nhead = len(layer._head_geoms)
        c.head = buf(np.array(layer._head_geoms or [0], dtype=np.int32), "int[]")
        c.njack = len(layer._jack_gids)
        c.jack_gids = buf(np.array(layer._jack_gids, dtype=np.int32), "int[]")
        c.q_rest, c.drown_delay = drives.Q_REST, drives.DROWN_DELAY_S
        c.tau_wet, c.tau_dry = drives.WET_TAU_WET_S, drives.WET_TAU_DRY_S
        c.partners = buf(np.zeros(m.ngeom, dtype=np.uint8), "uint8_t[]")
        self.s = f.new("jn_dstate*")

    def load(self) -> None:
        L, s = self.layer, self.s
        for k in self._F:
            setattr(s, k, getattr(L, "_" + k))
        s.w, s.t, s.n_onsets = L.state.w, L.t, L.n_onsets
        s.touching_world = int(L._touching_world)
        s.has_prev_speed = int(L._prev_speed is not None)
        s.prev_speed = 0.0 if L._prev_speed is None else L._prev_speed
        s.has_submerged = int(L._submerged_since is not None)
        s.submerged_since = (0.0 if L._submerged_since is None
                             else L._submerged_since)
        s.wet_in = int(getattr(L, "_wet_in", False))
        for k, n in enumerate(self.names):
            self.respawn_at[k] = L._respawn_at[n]
            self.ate_total[k] = L.ate_total[n]

    def substep(self, data, dt: float) -> None:
        self.lib.jn_drives_substep(data._address, self.cfg, self.s, dt)

    def diff(self) -> list:
        L, s, bad = self.layer, self.s, []
        for k in self._F:
            if getattr(s, k) != getattr(L, "_" + k):
                bad.append(k)
        if s.w != L.state.w: bad.append("w")
        if s.t != L.t: bad.append("t")
        if s.n_onsets != L.n_onsets: bad.append("n_onsets")
        if bool(s.touching_world) != bool(L._touching_world):
            bad.append("touching_world")
        if (L._prev_speed is None) == bool(s.has_prev_speed) or (
                L._prev_speed is not None and s.prev_speed != L._prev_speed):
            bad.append("prev_speed")
        if (L._submerged_since is None) == bool(s.has_submerged) or (
                L._submerged_since is not None
                and s.submerged_since != L._submerged_since):
            bad.append("submerged_since")
        if bool(s.wet_in) != bool(L._wet_in): bad.append("wet_in")
        for k, n in enumerate(self.names):
            if self.respawn_at[k] != L._respawn_at[n]:
                bad.append("respawn_at:" + n)
            if int(self.ate_total[k]) != L.ate_total[n]:
                bad.append("ate_total:" + n)
        return bad


class NativeWater:
    """C twin of `playground.Water.apply` for ONE Water (piece 3 of 4).

    Config (radii, masses, dry-geom mask, pool box, coefficients) and the
    additive-write state (`_last`, `_written`, `_last_any`) are copied from
    the Python object by `load()`; `diff()` compares the state exactly.
    `fma_norm=False` is the identity test's control only."""

    def __init__(self, water, fma_norm: bool = True):
        import playground as pg
        self.ffi, self.lib = build()
        f, m = self.ffi, water.model
        self.water = water
        keep = self._keep = []

        def buf(a, ct):
            keep.append(a)
            return f.from_buffer(ct, a)
        dry = np.zeros(m.ngeom, dtype=np.uint8)
        dry[sorted(water._dry_geoms)] = 1
        self.last = np.zeros(m.nbody * 3)
        self.written = np.zeros(m.nbody * 3)
        w = self.cfg = f.new("jn_water*")
        w.nbody, w.ngeom, w.fma_norm = m.nbody, m.ngeom, int(bool(fma_norm))
        w.radius = buf(np.ascontiguousarray(water._radius, dtype=float), "double[]")
        w.mass = buf(np.ascontiguousarray(m.body_mass, dtype=float), "double[]")
        w.dry = buf(dry, "uint8_t[]")
        w.geom_body = buf(np.ascontiguousarray(m.geom_bodyid, dtype=np.int32), "int[]")
        w.rho, w.g = pg.WATER_DENSITY, pg.GRAVITY
        w.last, w.written = buf(self.last, "double[]"), buf(self.written, "double[]")
        w.newf = buf(np.zeros(m.nbody * 3), "double[]")
        w.held = buf(np.zeros(m.nbody, dtype=np.uint8), "uint8_t[]")
        self.load()

    def load(self) -> None:
        W, w = self.water, self.cfg
        w.enabled = int(bool(W.enabled))
        w.x, w.y, w.half, w.surf = W.x, W.y, W.half, W.surface_z
        w.cd, w.c_lin = W.cd, W.c_lin
        self.last[:] = np.asarray(W._last, dtype=float).ravel()
        self.written[:] = np.asarray(W._written, dtype=float).ravel()
        w.last_any = int(bool(W._last_any))

    def apply(self, data) -> None:
        self.lib.jn_water_apply(data._address, self.cfg)

    def diff(self) -> list:
        W, bad = self.water, []
        if not np.array_equal(self.last, np.asarray(W._last).ravel()):
            bad.append("last")
        if not np.array_equal(self.written, np.asarray(W._written).ravel()):
            bad.append("written")
        if bool(self.cfg.last_any) != bool(W._last_any):
            bad.append("last_any")
        return bad


class NativeSynth:
    """C twin of `fast_sim._lazy_step` for ONE ContactAudioSynth (piece 4).

    C owns the per-sub-step pair grouping and the `_prev_pairs` /
    `_last_fired` state; for each pair that fires, `step()` gathers its
    contact indices and builds the event with the SAME Python expressions
    as the shipped step (mj_contactForce, np.mean of positions, sum of
    forces, `_make_event`) into `self.events`. `load()` copies state in,
    `diff()` compares it exactly."""

    MAXFIRE = 256

    def __init__(self, synth):
        from ContactAudio import REFRACTORY_S
        self.ffi, self.lib = build()
        f, m = self.ffi, synth.model
        self.synth, self.ng = synth, m.ngeom
        n2 = m.ngeom * m.ngeom
        self.prev_mark = np.zeros(n2, dtype=np.uint8)
        self.last = np.full(n2, -1e9)
        self.prev = np.zeros(n2, dtype=np.int32)
        self.cur = np.zeros(n2, dtype=np.int32)
        self.seen = np.zeros(n2, dtype=np.int32)
        self.fire = np.zeros(self.MAXFIRE, dtype=np.int32)
        b = f.from_buffer
        s = self.cfg = f.new("jn_synth*")
        s.ngeom, s.refr = m.ngeom, REFRACTORY_S
        s.prev_mark, s.last = b("uint8_t[]", self.prev_mark), b("double[]", self.last)
        s.prev, s.cur = b("int[]", self.prev), b("int[]", self.cur)
        s.seen = b("int[]", self.seen)
        self._pf = b("int[]", self.fire)
        self.events = []
        self.load()

    def load(self) -> None:
        S, ng = self.synth, self.ng
        self.prev_mark[:] = 0
        self.last[:] = -1e9
        keys = [a * ng + b for a, b in S._prev_pairs]
        for j, k in enumerate(keys):
            self.prev_mark[k] = 1
            self.prev[j] = k
        self.cfg.nprev = len(keys)
        for (a, b), t in S._last_fired.items():
            self.last[a * ng + b] = t

    def step(self, data) -> None:
        import mujoco
        nf = self.lib.jn_synth_step(data._address, self.cfg, self._pf, self.MAXFIRE)
        assert nf <= self.MAXFIRE
        if not nf:
            return
        S, ng, t = self.synth, self.ng, float(data.time)
        g = data.contact.geom[:int(data.ncon)].tolist()
        for key in self.fire[:nf].tolist():
            pair = (key // ng, key % ng)
            pts = []
            for i, (a, b) in enumerate(g):
                if ((a, b) if a <= b else (b, a)) != pair:
                    continue
                con = data.contact[i]
                f6 = np.zeros(6)
                mujoco.mj_contactForce(S.model, data, i, f6)
                pts.append((np.array(con.pos), abs(float(f6[0]))))
            pos = np.mean([p for p, _ in pts], axis=0)
            force = float(sum(f for _, f in pts))
            self.events.append(S._make_event(t, pair, pos, force))

    def diff(self) -> list:
        S, ng, bad = self.synth, self.ng, []
        mine = {(int(k) // ng, int(k) % ng) for k in self.prev[:self.cfg.nprev]}
        if mine != S._prev_pairs:
            bad.append("prev_pairs")
        lf = {(int(k) // ng, int(k) % ng): float(self.last[k])
              for k in np.flatnonzero(self.last != -1e9)}
        if lf != S._last_fired:
            bad.append("last_fired")
        return bad


def _main() -> None:
    from experiments.jack_env import JackEnv
    env = JackEnv(seed=153, lazy_contact_audio=True)
    env.reset(seed=153)
    w = env.w
    nc = NativeContacts(w._body_mask, w._ground_mask)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.uniform(-1, 1, env.action_space.shape))
    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        w._grounded()
    tp = (time.perf_counter() - t0) / N * 1e6
    t0 = time.perf_counter()
    for _ in range(N):
        nc.grounded(w.data)
    tc = (time.perf_counter() - t0) / N * 1e6
    print(f"ncon={w.data.ncon} _grounded python {tp:.2f} us, C {tc:.2f} us"
          f" -> x40 saves {(tp - tc) * 40 / 1000:.3f} ms a decision")


if __name__ == "__main__":
    _main()


# ── J6 (2b) wiring: one option, all four pieces ─────────────────────────
def _drives_store(nd) -> None:
    """Write the C drive state back into the Python layer (inverse of load)."""
    L, s = nd.layer, nd.s
    for k in nd._F:
        setattr(L, "_" + k, getattr(s, k))
    L.state.w, L.t, L.n_onsets = s.w, s.t, s.n_onsets
    L._touching_world = bool(s.touching_world)
    L._prev_speed = s.prev_speed if s.has_prev_speed else None
    L._submerged_since = s.submerged_since if s.has_submerged else None
    L._wet_in = bool(s.wet_in)
    for k, n in enumerate(nd.names):
        L._respawn_at[n] = float(nd.respawn_at[k])
        L.ate_total[n] = int(nd.ate_total[k])


def _water_store(nw) -> None:
    W = nw.water
    W._last = nw.last.reshape(np.shape(W._last)).copy()
    W._written = nw.written.reshape(np.shape(W._written)).copy()
    W._last_any = bool(nw.cfg.last_any)


def _synth_store(ns) -> None:
    S, ng = ns.synth, ns.ng
    S._prev_pairs = {(int(k) // ng, int(k) % ng)
                     for k in ns.prev[:ns.cfg.nprev]}
    S._last_fired = {(int(k) // ng, int(k) % ng): float(ns.last[k])
                     for k in np.flatnonzero(ns.last != -1e9)}


def _native_decide(self, action) -> None:
    """W0.decide with its 40-sub-step body run by the four C twins. Python
    state is loaded into C at the top of each decision and written back at
    the bottom, so everything outside the loop (decide(), death, respawn,
    observe) reads exactly the objects it reads today."""
    from experiments import drives as _drv
    from experiments.w0 import SUBSTEPS
    nat = self._native
    a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
    if a.shape[0] != self.action_dim:
        raise RuntimeError(f"action is {a.shape[0]}, not {self.action_dim}")
    gear = self.drives.gear_scale()
    lo = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=float)
    hi = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=float)
    ctrl = lo + (a[:6] * 0.5 + 0.5) * (hi - lo)
    ctrl[:4] *= gear
    force = a[6:8] * self.pg.ROVER_DRIVE_FORCE * gear
    self.drives.begin_decision()
    dt = float(self.model.opt.timestep)
    nd, nw, ns, nc = nat["drives"], nat["water"], nat["synth"], nat["contacts"]
    nd.load()
    if nw is not None:
        nw.load()
    ns.load()
    ns.events = []
    d, m, step, bid = self.data, self.model, self.mujoco.mj_step, self.rover_bid
    n_gated = 0
    for _ in range(SUBSTEPS):
        d.ctrl[:] = ctrl
        gate = nc.grounded(d)
        n_gated += int(gate)
        d.xfrc_applied[bid, :2] = force if gate else 0.0
        if nw is not None:
            nw.apply(d)
        step(m, d)
        ns.step(d)
        nd.substep(d, dt)
    _drives_store(nd)
    if nw is not None:
        _water_store(nw)
    _synth_store(ns)
    self.synth.events.extend(ns.events)
    d.xfrc_applied[bid, :2] = 0.0
    self.mujoco.mj_rnePostConstraint(m, d)
    self._prev_drive = _drv.DriveState(**vars(self.drives.state))
    self.drives.decide()
    self._audio = self._drain_audio()
    self.decisions += 1
    self.drive_gate_open += n_gated
    self.sim_seconds += SUBSTEPS * dt
    self.died_this_decision = False
    self.dying_obs = None
    if self.lethal:
        cause = self.death_cause()
        if cause:
            self._die(cause)


def apply_native_substep(w, fma_norm: bool = True) -> None:
    """Patch one W0 instance (the fast_sim way; w0.py untouched). Requires
    lazy_contact_audio semantics (piece 4 is the twin of `_lazy_step`, which
    is proven identical to the shipped step)."""
    import types
    w._native = {
        "contacts": NativeContacts(w._body_mask, w._ground_mask),
        "drives": NativeDrives(w.drives, fma_norm=fma_norm),
        "water": (NativeWater(w.water, fma_norm=fma_norm)
                  if w.water is not None else None),
        "synth": NativeSynth(w.synth),
    }
    w.decide = types.MethodType(_native_decide, w)
