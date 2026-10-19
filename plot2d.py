import pickle, numpy as np, matplotlib, glob
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("plot1d.py").read().split("C = {")[0].split("import matplotlib.pyplot as plt")[1])  # 同じ rcParams
K = ["reg", "mix", "quart", "cross"]
C = {"reg": "#2a78d6", "mix": "#eb6834", "quart": "#1baf7a", "cross": "#6250d6"}
LAB = {"reg": "$w_1^2+w_2^2$ (λ=1, m=1)", "mix": "$w_1^2+w_2^4$ (λ=3/4, m=1)",
       "quart": "$(w_1^2+w_2^2)^2$ (λ=1/2, m=1)", "cross": "$w_1^2w_2^2$ (λ=1/2, m=2)"}
fr = pickle.load(open("fric2d.pkl", "rb"))
R = {k: pickle.load(open(f"res2d_{k}.pkl", "rb")) for k in K}
fig, ax = plt.subplots(1, 4, figsize=(16, 3.9))
for k in K:
    runs = R[k]["runs"]; taus = np.array(sorted(runs))
    fin = [{q: runs[t][q][-1] for q in runs[t]} for t in taus]
    W = np.array([f["W"] for f in fin]); dF = np.array([f["dFeq"] for f in fin]); KL = np.array([f["KL"] for f in fin])
    S = np.array([f["Sig"] for f in fin]); W2 = np.array([f["W2"] for f in fin])
    ax[0].semilogx(taus, (dF + KL + W2**2 / taus) / W, "-o", ms=4, c=C[k], label=LAB[k])
    ax[1].loglog(taus, S * taus, "-o", ms=4, c=C[k]); ax[1].axhline(fr[k]["Igeo"], c=C[k], ls=":", lw=1.3)
    ax[2].semilogx(taus, W2**2 / (taus * S), "-o", ms=4, c=C[k])
    ax[3].semilogx(fr[k]["ns"], fr[k]["var"] / fr[k]["lam"], c=C[k])
x = np.log(fr["cross"]["ns"])
ax[3].semilogx(fr["cross"]["ns"], 1 - 2 / x - 2 / x**2, ":", c=C["cross"], lw=1.2)
ax[0].set(xlabel="学習時間 τ", ylabel="η", title="(a) 全体の飽和度\n$(ΔF+KL+W_2^2/τ)/W$", ylim=(0, 1.02))
ax[1].set(xlabel="学習時間 τ", ylabel="Σ·τ", title="(b) 散逸 Σ·τ\n点線: ゆっくり極限 log N ∫ζ dx")
ax[2].set(xlabel="学習時間 τ", ylabel="$η_Σ=W_2^2/(τΣ)$", title="(c) 速度限界の飽和度", ylim=(0, 1))
ax[3].set(xlabel="n", ylabel="Var(nL)/λ", title="(d) 熱容量/λ\n点線: 1−(m−1)(1/log n+1/log²n)/λ", ylim=(0, 1.1))
ax[0].legend(fontsize=8, loc="lower right")
fig.suptitle("図4  2D, N=400:  同じ λ=1/2 で重複度 m が違うと散逸と飽和度が変わる", x=0.01, ha="left")
fig.tight_layout(); fig.savefig("figs/fig4_2d.png", dpi=150)
for k in K:
    print(k, "Στ(τ=300)=%.4f theory=%.4f" % (R[k]["runs"][300.0]["Sig"][-1]*300, fr[k]["Igeo"]))
