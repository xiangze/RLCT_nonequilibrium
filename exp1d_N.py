import numpy as np, pickle
from tsl_slt import *
g = Grid(1, 40000)
res = {}
for nm, f, lam in [("w^2", lambda w: w**2, 1/2), ("w^4", lambda w: w**4, 1/4), ("w^6", lambda w: w**6, 1/6)]:
    M = Model(nm, g, f, lam, 1)
    Ns = np.geomspace(1e1, 1e6, 11)
    nsall = np.geomspace(1, 1e6, 400)
    fr = np.array([friction_1d(M, n) for n in nsall])
    x = np.log(nsall)
    rows = []
    p1 = M.pi(1)
    for N in Ns:
        k = x <= np.log(N) + 1e-9
        I = np.log(N) * np.trapezoid(fr[k, 0], x[k])
        Lth2 = np.trapezoid(np.sqrt(fr[k, 0]), x[k]) ** 2
        W2 = w2_1d(g.x1, p1, M.pi(N))
        lx = lam * np.log(N)
        rows.append((N, I, Lth2, W2**2, W2**2 / I, 2 * np.tanh(lx / 2) / lx))
    # τ_c の冪: n^{-2λ} を検証
    sl = np.polyfit(x[x > np.log(1e3)], np.log(fr[x > np.log(1e3), 2]), 1)[0]
    res[nm] = dict(lam=lam, rows=np.array(rows), ns=nsall, zeta=fr[:, 0], var=fr[:, 1], tc=fr[:, 2], tc_slope=sl)
    print(nm, "tau_c ~ n^%.4f (予測 -2λ = %.4f)" % (sl, -2 * lam), " Var(nL) at 1e6 = %.5f" % fr[-1, 1])
    for r in rows[::2]:
        print("   N=%8.0f  Στ_slow=%.4f ℒ²=%.4f W2²=%.4f  η_Σ=%.3f  2tanh(λlogN/2)/(λlogN)=%.3f" % r)
pickle.dump(res, open("res1d_N.pkl", "wb"))
