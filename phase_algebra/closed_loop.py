#!/usr/bin/env python
"""P2: 4 行列を合成して閉ループ行列 T を作り、スペクトル半径 ρ(T) で安定性を判定する。

    身体   x̄[k]   = F x̄[k−1] + G1 ū[k−1] + G0 ū[k]          (P1 で同定)
    筋     u_end[k] = a_tw u_end[k−1] + (1−a_tw) u_cmd[k]      (単収縮 τ=4.25ms)
           ū[k]    = b_tw u_end[k−1] + (1−b_tw) u_cmd[k]       (羽ばたき内平均)
    感覚   ω_f[k]  = a_s ω_f[k−1] + (1−a_s) Gs ω̄[k−1]         (統合36d: τ=12ms, ゲイン)
           ω̂      = ω_f[k−1−d]                               (純遅延 d 羽ばたき)
    制御   u_cmd[k] = D K y[k−1] ,  D = 0.35·diag(1−tanh²(K y_ref+b))  (動作点で線形化)

閉ループ状態 s = [x̄, u_end, ū, ω_f 履歴 (d+1), v_f] の一段写像 T を列ごとに構成し、
ρ(T) < 1 ⇔ ホバリング安定。遅延 d を振って、実験ログの「遅延の壁」と突き合わせる。
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PL = np.load("phase_algebra/outputs/plant_FG.npz")
F, G1, G0, XB_REF, PER = PL["F"], PL["G1"], PL["G0"], PL["xb_ref"], float(PL["per"])
SF = json.load(open("outputs/sensor_fit.json"))
GS = np.array([SF[a]["G"] for a in "xyz"])
TAU_S = SF["x"]["tau_ms"] / 1000.0
TAU_TW, TAU_VIS = 0.00425, 0.030
A_TW = np.exp(-PER / TAU_TW)                      # 羽ばたき末の筋状態の残存
B_TW = (TAU_TW / PER) * (1 - np.exp(-PER / TAU_TW))   # 羽ばたき内平均に残る前状態の割合
A_S = np.exp(-PER / TAU_S)
A_V = np.exp(-PER / TAU_VIS)
VIS_GAIN = 0.63 * 1.2                             # 統合37 の視覚模型の小信号ゲイン
Z_T = 12.0
N_X, N_U = 7, 5


def load_K(name):
    if name.endswith(".npy"):
        th = np.load(name); th = th[12:]
    else:
        path, seed = name.split(":")
        res = json.load(open(path)); th = np.array([r for r in res if r["seed"] == int(seed)][0]["theta"])
    return th[:N_U * N_X].reshape(N_U, N_X), th[N_U * N_X:]


def y_of(xb, om_hat, eb):
    """制御則への入力 (bioflight と同じ正規化)。xb は絶対状態"""
    return np.array([(Z_T - xb[5]) / 5.0, -xb[6] / 30.0, eb[0], eb[1], om_hat[0] / 20, om_hat[1] / 20, om_hat[2] / 20])


def build_T(K, b, d=0, vision=False, sensor="circuit"):
    nx, nu = 9, 5
    nh = d + 1
    nv = 2 if vision else 0
    N = nx + nu + nu + 3 * nh + nv
    # 動作点での tanh 微分
    om_ref = GS * XB_REF[2:5] if sensor == "circuit" else XB_REF[2:5]
    eb_ref = VIS_GAIN * XB_REF[:2] if vision else XB_REF[:2]
    y_ref = y_of(XB_REF, om_ref, eb_ref)
    D = 0.35 * np.diag(1 - np.tanh(K @ y_ref + b) ** 2)
    KD = D @ K

    def step(s):
        xb = s[:nx]; u_end = s[nx:nx + nu]; ubar_prev = s[nx + nu:nx + 2 * nu]
        hist = s[nx + 2 * nu:nx + 2 * nu + 3 * nh].reshape(nh, 3)   # [ω_f[k−1], …, ω_f[k−1−d]]
        v_f = s[nx + 2 * nu + 3 * nh:]
        om_hat = hist[d]                                          # 遅延 d
        eb = v_f if vision else xb[:2]
        dy = np.array([-xb[5] / 5.0, -xb[6] / 30.0, eb[0], eb[1], om_hat[0] / 20, om_hat[1] / 20, om_hat[2] / 20])
        du_cmd = KD @ dy
        u_end_n = A_TW * u_end + (1 - A_TW) * du_cmd
        ubar = B_TW * u_end + (1 - B_TW) * du_cmd
        xb_n = F @ xb + G1 @ ubar_prev + G0 @ ubar
        if sensor == "circuit":
            om_f_n = A_S * hist[0] + (1 - A_S) * GS * xb[2:5]
        else:                                                     # 真の遅いω: 遅れなし
            om_f_n = xb[2:5]
        hist_n = np.vstack([om_f_n[None, :], hist[:-1]]) if nh > 1 else om_f_n[None, :]
        v_f_n = A_V * v_f + (1 - A_V) * VIS_GAIN * xb[:2] if vision else v_f
        return np.concatenate([xb_n, u_end_n, ubar, hist_n.ravel(), v_f_n])

    T = np.zeros((N, N))
    for i in range(N):
        e = np.zeros(N); e[i] = 1.0
        T[:, i] = step(e)
    return T


def analyse(T):
    """(全体ρ, 速い姿勢モード |λ|, その周期ms, 倍加ms)。速い = 振動周期 < 300ms。
    全体ρが1をわずかに越えるのは位置・高度の遅いドリフトモードで、飛行を殺すのは
    速い姿勢モードが単位円を越える点 (hover_growth.py で検証)。"""
    ev = np.linalg.eigvals(T)
    rho = max(abs(ev))
    fast = [l for l in ev if 1e-3 < abs(np.angle(l)) < np.pi - 1e-3
            and 2 * np.pi / abs(np.angle(l)) * PER < 0.3]
    if not fast:
        return rho, np.nan, np.inf, np.inf
    lam = max(fast, key=abs)
    per_ms = 2 * np.pi / abs(np.angle(lam)) * PER * 1000
    tau = np.log(2) / abs(np.log(abs(lam))) * PER * 1000 if abs(abs(lam) - 1) > 1e-9 else np.inf
    return rho, abs(lam), per_ms, tau


CONTROLLERS = [
    ("ES-K 統合31 (bioflight_best)", "outputs/bioflight_best.npy", False,
     "安価ループ: d0 3.00s, d1 3.00s, d2 2.35s(劣化), d3 0.96s → 壁 2-3"),
    ("ES 遅延2訓練 (es_takeoff_best)", "outputs/es_takeoff_best.npy", False,
     "d2 6.00s, d3 0.93s, d4 0.67s → 壁 2-3"),
    ("ES 遅延3訓練 (es_takeoff_d3_best)", "outputs/es_takeoff_d3_best.npy", False,
     "d3 6.00s, d4 2.37s → 壁 3-4"),
    ("報酬学習 遅延2訓練 個体2 (v6)", "outputs/reward_K_v6_delay2.json:2", False,
     "模型 d2 6/6 完走、実回路 10.00s×4/4 → 壁 ≥2"),
    ("報酬学習 視覚+遅延2訓練 個体4", "outputs/reward_K.json:4", True,
     "視覚遅れ+d2 で 6/6 完走 → 壁 ≥2"),
]


def main():
    print(f"羽ばたき周期 {PER*1000:.3f} ms / 筋 a_tw={A_TW:.3f} b_tw={B_TW:.3f} / 感覚 a_s={A_S:.3f} Gs={np.round(GS,2)}")
    print("開ループ (制御なし) の ρ:", f"{analyse(F)[0]:.4f}  (P1 の不安定モード)\n")
    out = {}
    for name, src, vis, empirical in CONTROLLERS:
        K, b = load_K(src)
        rows = []
        line = f"{name}\n  実験ログ: {empirical}\n  予測 ρ(T):"
        rho_true, fast_true, _, _ = analyse(build_T(K, b, 0, vis, sensor="true"))
        line += f"  [真の遅いω] 速い={fast_true:.4f} 全体={rho_true:.4f}"
        for d in range(0, 6):
            rho, fast, per_ms, tau = analyse(build_T(K, b, d, vis, sensor="circuit"))
            rows.append(dict(d=d, rho=rho, fast=fast, period_ms=per_ms, doubling_ms=tau))
            line += f"\n    d{d}: 速い姿勢モード |λ|={fast:.4f} (周期{per_ms:.0f}ms, 倍加{tau:.0f}ms)  全体ρ={rho:.4f}"
        wall = next((r["d"] for r in rows if r["fast"] > 1.0), None)
        line += f"\n  → 予測される壁: {'d=%d で速い姿勢モードが不安定化' % wall if wall is not None else 'd5 まで速いモードは安定'}"
        line += f"\n  → 遅いドリフト (位置・高度): 倍加 {np.log(2)/np.log(rows[0]['rho'])*PER*1000/1000 if rows[0]['rho']>1 else float('inf'):.1f} s"
        print(line + "\n")
        out[name] = dict(source=src, vision=vis, empirical=empirical, rho_true=rho_true, fast_true=fast_true, rows=rows, wall=wall)
    json.dump(out, open("phase_algebra/outputs/closed_loop.json", "w"), indent=1, ensure_ascii=False)
    print("保存: phase_algebra/outputs/closed_loop.json")


if __name__ == "__main__":
    main()
