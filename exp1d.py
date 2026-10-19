import numpy as np, pickle, time
from tsl_slt import *
g = Grid(1, 2000)
N = 1e4
models = [Model("w^2", g, lambda w: w**2, 1/2, 1),
          Model("w^4", g, lambda w: w**4, 1/4, 1),
          Model("w^6", g, lambda w: w**6, 1/6, 1)]
taus = np.geomspace(1e-2, 1e4, 13)
out = {}
for M in models:
    t0 = time.time()
    # 摩擦 ζ(x) とそこから thermodynamic length (幾何スケジュールの場合)
    ns = np.geomspace(1, N, 200)
    fr = np.array([friction_1d(M, n) for n in ns])
    x = np.log(ns)
    zeta = fr[:, 0]
    Lth = np.trapezoid(np.sqrt(zeta), x)           # 最適スケジュールでの ℒ
    Igeo = np.log(N) * np.trapezoid(zeta, x)       # 幾何スケジュールの Στ (ゆっくり極限)
    finals, runs = [], {}
    for tau in taus:
        r = run(M, N, tau, steps=(4000 if tau < 30 else 40000), n_out=80)
        finals.append({k: r[k][-1] for k in r})
        if tau in taus[[2, 6, 10]]:
            runs[tau] = r
    # 情報獲得の平衡値と瞬間 λ_eq(n)
    info_eq = np.array([-M.logZ(n) + M.logZ(1) - n*np.dot(M.pi(n), M.L) + np.dot(M.pi(1), M.L) for n in ns])
    out[M.name] = dict(lam=M.lam, ns=ns, zeta=zeta, var=fr[:, 1], tc=fr[:, 2], Lth=Lth, Igeo=Igeo,
                       taus=taus, finals=finals, runs=runs, info_eq=info_eq,
                       dFeq_N=-M.logZ(N) + M.logZ(1))
    print(M.name, "ℒ²=%.4f  geo slow Στ=%.4f  ΔF=%.3f  (%.0fs)" % (Lth**2, Igeo, out[M.name]['dFeq_N'], time.time()-t0))
pickle.dump(out, open("res1d.pkl", "wb"))
