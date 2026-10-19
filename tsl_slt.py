"""
有限時間 Landauer / 熱力学的速度限界 (TSL) と RLCT — 最小の玩具モデル

設定 (T = 1, 移動度 1):
  事前分布 φ = 箱 [-a,a]^d 上の一様分布、損失 L(w) は realizable (min L = 0)。
  有効サンプル数 n(t) を n0 -> N に上げる (データを順に入れる) と、
  ポテンシャル U_t = n(t) L が時間依存し、密度 q_t は Fokker-Planck に従う。

  厳密な収支 (状態関数だけで書ける):
      W(t) = ΔF_eq(t) + KL(q_t || π_{n(t)}) + Σ(t)
      ΔF_eq(t) = -log Z(n(t)) + log Z(n0) ≈ λ log(n/n0) - (m-1) loglog n + ...
  速度限界:
      Σ(t) = ∫ |q̇|_W² dt  >=  L(t)²/t  >=  W2(q_0, q_t)²/t
  (過減衰 Langevin の速度場 v = -∇(U + log q) は勾配場なので、
   瞬間的な Benamou-Brenier は等号。たるみは「速さの不均一」と「測地線からのずれ」だけ。)

離散化: 格子上の可逆マスター方程式 (k_{i->j} = dx^-2 exp(-(U_j-U_i)/2))。
  1 ステップ = (1) n を更新 (仕事が入る、エントロピー不変)
               (2) 固定ポテンシャルで後退 Euler 緩和 (これ自体が π を不変にする可逆な
                   確率行列なので KL は単調減少 -> 離散でも Σ >= 0 が厳密)。
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class Grid:
    def __init__(self, d, M, a=1.0):
        self.d, self.M, self.a = d, M, a
        self.dx = 2 * a / M
        x = -a + self.dx * (np.arange(M) + 0.5)
        self.x1 = x
        if d == 1:
            self.X = [x]
        else:
            X1, X2 = np.meshgrid(x, x, indexing="ij")
            self.X = [X1.ravel(), X2.ravel()]
        self.K = M ** d
        self.cell = self.dx ** d
        self.logphi_cell = np.log(self.cell / (2 * a) ** d)   # log(φ_i dx^d)
        # 隣接辺 (i, j)
        idx = np.arange(self.K).reshape((M,) * d)
        I, J = [], []
        for ax in range(d):
            sl0 = [slice(None)] * d
            sl1 = [slice(None)] * d
            sl0[ax] = slice(0, M - 1)
            sl1[ax] = slice(1, M)
            I.append(idx[tuple(sl0)].ravel())
            J.append(idx[tuple(sl1)].ravel())
        self.ei = np.concatenate(I)
        self.ej = np.concatenate(J)


class Model:
    def __init__(self, name, grid, Lfun, lam, m):
        self.name, self.g, self.lam, self.m = name, grid, lam, m
        self.L = Lfun(*grid.X)

    # ---- 平衡量 ----
    def logZ(self, n):
        a = -n * self.L + self.g.logphi_cell
        mx = a.max()
        return mx + np.log(np.exp(a - mx).sum())

    def pi(self, n):
        a = -n * self.L
        p = np.exp(a - a.max())
        return p / p.sum()

    # ---- 生成子 ----
    def relax_matrix(self, n, dt):
        g = self.g
        U = n * self.L
        dU = U[g.ej] - U[g.ei]
        kij = np.exp(-0.5 * dU) / g.dx ** 2   # i -> j
        kji = np.exp(+0.5 * dU) / g.dx ** 2   # j -> i
        # G[j,i] = rate i->j  (列和 0、p は列ベクトル)
        rows = np.concatenate([g.ej, g.ei])
        cols = np.concatenate([g.ei, g.ej])
        vals = np.concatenate([kij, kji])
        G = sp.csc_matrix((vals, (rows, cols)), shape=(g.K, g.K))
        out = np.asarray(G.sum(axis=0)).ravel()
        G = G - sp.diags(out)
        return (sp.identity(g.K, format="csc") - dt * G).tocsc()

    def relax_solve(self, p, n, dt):
        """(I - dt G) p_new = p を解く。1D は三重対角で高速に。"""
        g = self.g
        if g.d != 1:
            return spla.spsolve(self.relax_matrix(n, dt), p)
        from scipy.linalg import solve_banded
        dU = n * (self.L[1:] - self.L[:-1])
        c = dt / g.dx ** 2
        kr = c * np.exp(-0.5 * dU)   # i -> i+1
        kl = c * np.exp(+0.5 * dU)   # i+1 -> i
        diag = np.ones(g.K)
        diag[:-1] += kr
        diag[1:] += kl
        ab = np.zeros((3, g.K))
        ab[0, 1:] = -kl          # 上対角: p_{i+1} から i へ
        ab[1] = diag
        ab[2, :-1] = -kr         # 下対角: p_i から i+1 へ
        return solve_banded((1, 1), ab, p)

    def sigma(self, p, n):
        """瞬間のエントロピー生成率 (マスター方程式の厳密な式)。"""
        g = self.g
        dU = n * (self.L[g.ej] - self.L[g.ei])
        a = np.exp(-0.5 * dU) * p[g.ei]   # i->j の流れ (dx^-2 を除く)
        b = np.exp(+0.5 * dU) * p[g.ej]   # j->i
        ok = (a > 1e-300) & (b > 1e-300)
        return np.sum((a[ok] - b[ok]) * np.log(a[ok] / b[ok])) / g.dx ** 2

    # ---- 非平衡自由エネルギー  F(p; n) = n<L> + KL(p || φ) ----
    def F(self, p, n):
        pz = p[p > 0]
        info = np.sum(pz * (np.log(pz) - self.g.logphi_cell))
        return n * np.dot(p, self.L), info


def w2_1d(x, p, q):
    """1D の W2 (分位関数の差の L2)。"""
    cp, cq = np.cumsum(p), np.cumsum(q)
    u = np.linspace(0, 1, 4001)[1:-1]
    xp = np.interp(u, cp, x)
    xq = np.interp(u, cq, x)
    return np.sqrt(np.mean((xp - xq) ** 2))


def w2_2d(g, p, q, coarse=50):
    import ot
    f = g.M // coarse
    P = p.reshape(g.M, g.M).reshape(coarse, f, coarse, f).sum(axis=(1, 3)).ravel()
    Q = q.reshape(g.M, g.M).reshape(coarse, f, coarse, f).sum(axis=(1, 3)).ravel()
    xc = g.x1.reshape(coarse, f).mean(axis=1)
    X1, X2 = np.meshgrid(xc, xc, indexing="ij")
    Xc = np.stack([X1.ravel(), X2.ravel()], 1)
    keepP, keepQ = P > 1e-12, Q > 1e-12
    A, B = P[keepP] / P[keepP].sum(), Q[keepQ] / Q[keepQ].sum()
    C = ot.dist(Xc[keepP], Xc[keepQ])
    return np.sqrt(ot.emd2(A, B, C, numItermax=2_000_000))


def run(model, N, tau, n0=1.0, steps=4000, schedule="geom", n_out=60, w2=True):
    """n(t) を n0 -> N に時間 tau で上げる。running な収支を返す。"""
    g = model.g
    s = np.linspace(0, 1, steps + 1)
    if schedule == "geom":
        n_of = n0 * (N / n0) ** s
    else:  # 線形 (データが一定の速さで届く)
        n_of = n0 + (N - n0) * s
    dt = tau / steps
    p = model.pi(n0)           # 初期 = n0 の事後 (n0=1 でほぼ事前分布)
    p0 = p.copy()
    logZ0 = model.logZ(n0)
    E0, I0 = model.F(p, n0)
    W = 0.0
    sig_prev = model.sigma(p, n0)
    Sint = 0.0          # ∫σ dt (検算用)
    Lpath = 0.0         # ∫ sqrt(σ) dt
    Lmean_prev = np.dot(p, model.L)
    out_idx = set(np.unique(np.geomspace(1, steps, n_out).astype(int)))
    rec = []
    for k in range(1, steps + 1):
        n_old, n_new = n_of[k - 1], n_of[k]
        p = model.relax_solve(p, n_new, dt)
        p = np.clip(p, 0, None); p /= p.sum()
        Lmean = np.dot(p, model.L)
        W += (n_new - n_old) * 0.5 * (Lmean_prev + Lmean)  # 台形則 (2 次精度)
        Lmean_prev = Lmean
        sig = model.sigma(p, n_new)
        Sint += 0.5 * (sig + sig_prev) * dt
        Lpath += 0.5 * (np.sqrt(sig) + np.sqrt(sig_prev)) * dt
        sig_prev = sig
        if k in out_idx:
            t = k * dt
            E, I = model.F(p, n_new)
            dFeq = -model.logZ(n_new) + logZ0
            KL = (E + I) + model.logZ(n_new)               # F - F_eq
            if w2:
                W2 = w2_1d(g.x1, p0, p) if g.d == 1 else w2_2d(g, p0, p)
            else:
                W2 = np.nan
            rec.append(dict(t=t, n=n_new, W=W, dFeq=dFeq, KL=KL,
                            Sig=W - dFeq - KL, Sint=Sint, sig=sig,
                            L=Lpath, W2=W2, info=I - I0, energy=E - E0))
    return {k: np.array([r[k] for r in rec]) for k in rec[0]}


def friction_1d(model, n):
    """ゆっくり駆動の摩擦係数 ζ(x), x = log n。
    ζ = <δX, (-G)^{-1} δX>_π,  X = ∂U/∂x = n L。
    1D の可逆な birth-death 鎖では Poisson 方程式が閉形式で解ける:
        ζ = Σ_i F_i² / (π_i k_{i->i+1}),  F_i = Σ_{j<=i} π_j δX_j
    戻り値: (ζ, Var(nL) [= 熱容量 -> λ], 積分自己相関時間 τ_c = ζ/Var)
    """
    g = model.g
    pi = model.pi(n)
    X = n * model.L
    dX = X - np.dot(pi, X)
    w = pi * dX
    Fl = np.cumsum(w)[:-1]                 # 左から
    Fr = -np.cumsum(w[::-1])[::-1][1:]     # 右から (丸め誤差を裾で避ける)
    c = np.argmax(pi)
    F = np.where(np.arange(g.K - 1) < c, Fl, Fr)
    dU = n * (model.L[1:] - model.L[:-1])
    flux = pi[:-1] * np.exp(-0.5 * dU) / g.dx ** 2   # = π_i k_{i->i+1} (詳細釣り合いで対称)
    ok = flux > 1e-280
    zeta = np.sum(F[ok] ** 2 / flux[ok])
    var = np.dot(pi, dX ** 2)
    return zeta, var, zeta / var


def friction_2d(model, n):
    """ζ = <δX, (-G^T)^{-1} δX>_π を後ろ向き方程式で解く (π で割らない)。"""
    g = model.g
    pi = model.pi(n)
    X = n * model.L
    dX = X - np.dot(pi, X)
    GT = (-(model.relax_matrix(n, 1.0) - sp.identity(g.K))).T.tolil()  # G^T
    b = -dX.copy()
    c = int(np.argmax(pi))
    GT[c, :] = 0.0; GT[c, c] = 1.0; b[c] = 0.0     # 定数の自由度を固定
    h = spla.spsolve(GT.tocsc(), b)
    h -= np.dot(pi, h)
    zeta = np.dot(pi * dX, h)
    var = np.dot(pi, dX ** 2)
    return zeta, var, zeta / var
