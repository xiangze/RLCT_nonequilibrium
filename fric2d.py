import numpy as np, pickle
from tsl_slt import *
g = Grid(2, 160); N = 400.0
defs = {"reg":  ("w1^2+w2^2",      lambda a,b: a**2+b**2,      1.0, 1),
        "quart":("(w1^2+w2^2)^2",  lambda a,b: (a**2+b**2)**2, 0.5, 1),
        "cross":("w1^2 w2^2",      lambda a,b: a**2*b**2,      0.5, 2),
        "mix":  ("w1^2+w2^4",      lambda a,b: a**2+b**4,      0.75, 1)}
out = {}
for k, (nm, f, lam, m) in defs.items():
    M = Model(nm, g, f, lam, m)
    ns = np.geomspace(1, N, 40); x = np.log(ns)
    fr = np.array([friction_2d(M, n) for n in ns])
    W2 = w2_2d(g, M.pi(1), M.pi(N), coarse=80)
    I = np.log(N) * np.trapezoid(fr[:, 0], x); L2 = np.trapezoid(np.sqrt(fr[:, 0]), x) ** 2
    hi = x > np.log(50)
    sl = np.polyfit(x[hi], np.log(fr[hi, 2]), 1)[0]
    out[k] = dict(name=nm, lam=lam, m=m, ns=ns, zeta=fr[:, 0], var=fr[:, 1], tc=fr[:, 2],
                  Igeo=I, Lth2=L2, W2N=W2, tc_slope=sl, dFeq_N=-M.logZ(N) + M.logZ(1))
    print(f"{nm:15s} λ={lam} m={m}  Var(nL)@400={fr[-1,1]:.3f}  τ_c~n^{sl:.3f}  Στ_slow={I:.4f} ℒ²={L2:.4f} W2²={W2**2:.4f} η_Σ,slow={W2**2/I:.3f}  ΔF={out[k]['dFeq_N']:.3f}")
pickle.dump(out, open("fric2d.pkl", "wb"))
