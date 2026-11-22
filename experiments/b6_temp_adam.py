"""b6_temp_adam.py — row B-6 code diff, item (a) (KNOWN_ANSWERS B-6 addendum 11-22). Adam(lr 3e-4, b1 .9, b2 .999, eps 1e-8) on the two published temperature losses:
SBX 0.28.0 sac.py:359  L = log(a) * (H - tgt)   -> dL/dtheta = (H - tgt)
SimBa sac_update.py:153 L = a * (H - tgt)       -> dL/dtheta = a * (H - tgt)   (theta = log a, a = exp(theta))
Drive: (H - tgt) = d(t) given; report a after N updates. Pure arithmetic, no learner."""
import numpy as np
def run(kind, d, lr=3e-4, b1=.9, b2=.999, eps=1e-8):
    th, m, v = 0.0, 0.0, 0.0; out = []
    for t, g0 in enumerate(d, 1):
        a = np.exp(th); g = g0 if kind == "sbx" else a * g0
        m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g*g
        th -= lr * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps); out.append(np.exp(th))
    return np.array(out)
N = 300_000; rng = np.random.default_rng(0)
cases = {"const +1 (entropy above target)": np.ones(N),
         "noisy mean +0.2, sd 1": 0.2 + rng.standard_normal(N),
         "noisy mean 0, sd 1 (at target)": rng.standard_normal(N)}
for name, d in cases.items():
    s, b = run("sbx", d), run("simba", d)
    pts = [1_000, 10_000, 30_000, 100_000, 300_000]
    print(name); print("  updates  " + "  ".join(f"{p:>9}" for p in pts))
    print("  sbx      " + "  ".join(f"{s[p-1]:9.2e}" for p in pts))
    print("  simba    " + "  ".join(f"{b[p-1]:9.2e}" for p in pts))
    # first update at which alpha < 0.01
    for k, x in (("sbx", s), ("simba", b)):
        i = np.argmax(x < 0.01); print(f"  {k}: alpha<0.01 first at update {i+1 if x[i]<0.01 else 'never'}")
