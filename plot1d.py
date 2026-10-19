import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "lines.linewidth": 2, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "legend.frameon": False})
C = {"w^2": "#2a78d6", "w^4": "#eb6834", "w^6": "#1baf7a"}
LAB = {"w^2": "$w^2$ (λ=1/2)", "w^4": "$w^4$ (λ=1/4)", "w^6": "$w^6$ (λ=1/6)"}
o = pickle.load(open("res1d.pkl", "rb")); oN = pickle.load(open("res1d_N.pkl", "rb"))

# ---- Fig 1: τ スキャン ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for k, v in o.items():
    t = v["taus"]; F = v["finals"]
    W = np.array([f["W"] for f in F]); dF = np.array([f["dFeq"] for f in F])
    KL = np.array([f["KL"] for f in F]); S = np.array([f["Sig"] for f in F]); W2 = np.array([f["W2"] for f in F])
    L = np.array([f["L"] for f in F])
    ax[0].semilogx(t, (dF + KL + W2**2 / t) / W, "-o", ms=4, c=C[k], label=LAB[k])
    ax[0].semilogx(t, dF / W, "--", c=C[k], lw=1.3)
    ax[1].loglog(t, S * t, "-o", ms=4, c=C[k], label=LAB[k])
    ax[1].axhline(v["Igeo"], c=C[k], ls=":", lw=1.3)
    ax[2].semilogx(t, W2**2 / (t * S), "-o", ms=4, c=C[k], label=LAB[k])
ax[0].set(xlabel="学習時間 τ", ylabel="飽和度", title="(a) 実線: η = (ΔF+KL+$W_2$²/τ)/W\n破線: Landauer 比 ΔF/W")
ax[1].set(xlabel="学習時間 τ", ylabel="Σ·τ", title="(b) 散逸 Σ·τ\n点線: ゆっくり極限 log N ∫ζ dx ")
ax[2].set(xlabel="学習時間 τ", ylabel="η_Σ = $W_2$²/(τΣ)", title="(c) 速度限界 Σ ≥ $W_2$²/τ の飽和度", ylim=(0, 1))
ax[0].legend(loc="lower right"); ax[0].set_ylim(0, 1.02)
fig.suptitle("図1  1D, N=10⁴, 幾何スケジュール $n(t)=N^{t/\\tau}$ : βW ≥ ΔF_eq + KL + $W_2$²/τ の飽和", x=0.01, ha="left")
fig.tight_layout(); fig.savefig("figs/fig1_tau_scan_1d.png", dpi=150)

# ---- Fig 2: 学習途中 ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for k, v in o.items():
    taus = sorted(v["runs"])
    r = v["runs"][taus[2]]  # τ = 1000 (ゆっくり)
    ax[0].semilogx(r["n"], r["W2"]**2 / r["t"] / r["Sig"], c=C[k], label=LAB[k])
    # ζ からの予測 (ゆっくり極限、running)
    ns, z = v["ns"], v["zeta"]; x = np.log(ns)
    pred = []
    for nn in r["n"]:
        kk = x <= np.log(nn) + 1e-9
        I = np.log(nn) * np.trapezoid(z[kk], x[kk])
        pred.append(np.interp(nn, r["n"], r["W2"])**2 / I)
    ax[0].semilogx(r["n"], pred, ":", c="#0b0b0b", lw=1)
    for tau, ls in [(taus[0], "-"), (taus[1], "--")]:
        rr = v["runs"][tau]
        lamR = np.gradient(rr["info"], np.log(rr["n"]))
        ax[1 if k != "w^6" else 1].semilogx(rr["n"], lamR, ls, c=C[k], lw=1.6 if ls == "-" else 1.3)
    ax[1].axhline(v["lam"], c=C[k], ls=":", lw=1)
    r0 = v["runs"][taus[0]]
    ax[2].semilogx(r0["n"], (r0["dFeq"] + r0["KL"] + r0["W2"]**2 / r0["t"]) / r0["W"], c=C[k], label=LAB[k])
ax[0].set(xlabel="有効サンプル数 n(t)", ylabel="η_Σ(t)", title="(a) 学習途中の速度限界の飽和度 (τ=10³)\n点線: ζ からのゆっくり極限の予測", ylim=(0, 1.02))
ax[1].set(xlabel="有効サンプル数 n(t)", ylabel="λ_R(t) = d I / d log n", title="(b) 有限時間 RLCT  λ_R(t)\n実線 τ=0.1 / 破線 τ=10 / 点線 平衡 λ", ylim=(0, 1.15))
ax[2].set(xlabel="有効サンプル数 n(t)", ylabel="η(t)", title="(c) 速い学習 (τ=0.1) の全体の飽和度 η(t)", ylim=(0.6, 1.02))
ax[0].legend(loc="lower left")
fig.suptitle("図2  学習過程に沿った飽和度と有限時間 RLCT (1D, N=10⁴)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig("figs/fig2_running_1d.png", dpi=150)

# ---- Fig 3: 熱容量・緩和・N 依存 ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for k, v in oN.items():
    ax[0].semilogx(v["ns"], v["var"] / v["lam"], c=C[k], label=LAB[k])
    ax[1].loglog(v["ns"], v["tc"], c=C[k], label=LAB[k] + f"  傾き {v['tc_slope']:.3f}")
    R = v["rows"]
    ax[2].plot(v["lam"] * np.log(R[:, 0]), R[:, 4], "-o", ms=4, c=C[k], label=LAB[k])
lx = np.linspace(0.3, 7.5, 100)
ax[2].plot(lx, 2 * np.tanh(lx / 2) / lx, ":", c="#0b0b0b", lw=1.3, label="2 tanh(λlogN/2)/(λlogN)")
ax[0].set(xlabel="n", ylabel="Var(nL) / λ", title="(a) 熱容量 = RLCT:  Var(nL) → λ", ylim=(0.9, 1.05))
ax[1].set(xlabel="n", ylabel="積分自己相関時間 τ_c", title="(b) 緩和時間 τ_c ∝ $n^{-2\\lambda}$  (摩擦 ζ = λ τ_c)")
ax[2].set(xlabel="λ log N", ylabel="η_Σ (ゆっくり極限)", title="(c) 飽和度は主に λ log N で決まる", ylim=(0, 1.02))
for a in ax: a.legend(fontsize=8)
fig.suptitle("図3  1D:  ゆっくり駆動の散逸を決める静的量 (λ) と動的量 (τ_c)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig("figs/fig3_capacity_1d.png", dpi=150)
print("ok")
