import numpy as np, pickle, time, sys
from tsl_slt import *
which = sys.argv[1:]
g = Grid(2, 160)
N = 400.0
defs = {"reg":  ("w1^2+w2^2",      lambda a,b: a**2+b**2,      1.0, 1),
        "quart":("(w1^2+w2^2)^2",  lambda a,b: (a**2+b**2)**2, 0.5, 1),
        "cross":("w1^2 w2^2",      lambda a,b: a**2*b**2,      0.5, 2),
        "mix":  ("w1^2+w2^4",      lambda a,b: a**2+b**4,      0.75, 1)}
taus = [0.03, 0.3, 3.0, 30.0, 300.0]
import tsl_slt
tsl_slt.w2_2d.__defaults__ = (40,)
for key in which:
    nm, f, lam, m = defs[key]
    M = Model(nm, g, f, lam, m)
    t0 = time.time()
    ns = np.geomspace(1, N, 50)
    fr = np.array([friction_2d(M, n) for n in ns])
    x = np.log(ns)
    res = dict(name=nm, lam=lam, m=m, ns=ns, zeta=fr[:,0], var=fr[:,1], tc=fr[:,2],
               Lth=np.trapezoid(np.sqrt(fr[:,0]), x), Igeo=np.log(N)*np.trapezoid(fr[:,0], x),
               dFeq_N=-M.logZ(N)+M.logZ(1), runs={})
    print(nm, "friction done %.0fs" % (time.time()-t0), flush=True)
    for tau in taus:
        st = 3000 if tau >= 30 else 1500
        res["runs"][tau] = run(M, N, tau, steps=st, n_out=40)
        r = res["runs"][tau]
        print(f"  {nm} tau={tau}: Sig*tau={r['Sig'][-1]*tau:.4f} (slow theory {res['Igeo']:.4f}) etaSig={r['W2'][-1]**2/tau/r['Sig'][-1]:.3f} {time.time()-t0:.0f}s", flush=True)
    pickle.dump(res, open(f"res2d_{key}.pkl", "wb"))
