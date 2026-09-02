#!/usr/bin/env python
"""P4: 歩行を脚位相の結合振動子として解析する (複素コヒーレンス行列 + Kuramoto フィット)。

各脚 i の femur 角を帯域通過 (3–15 Hz) し、解析信号 (Hilbert) から連続位相 φ_i(t) を得る。
    Z_ij = ⟨ exp(i(φ_i − φ_j)) ⟩        (6×6 エルミート複素行列: 脚間の位相関係とロック強度)
Z の主固有ベクトルの偏角が集団歩容パターン (三脚なら LF,RM,LH 同相 / RF,LM,RH 逆相)。
続いて Kuramoto 型
    dφ_i/dt = ω_i + Σ_j [a_ij sin(φ_j−φ_i) + b_ij cos(φ_j−φ_i)]
を最小二乗でフィットし、ロック状態まわりのヤコビアン固有値で歩容の線形安定性を評価する。
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

Z = np.load(sys.argv[1] if len(sys.argv) > 1 else "phase_algebra/outputs/walk_timeseries.npz")
dt = float(Z["dt"]); LEGS = [str(x) for x in Z["legs"]]
C = Z["contact"]; Q = Z["q"]
# 直立区間のみ使う (転倒後の位相関係は無意味): 胸の高さ z>0.10 が続く先頭区間
zc_ = Q[:, 14]; bad = np.where(zc_ < 0.10)[0]; T_ok = int(bad[0]) if len(bad) else len(Q)
Q = Q[:T_ok]; C = C[:T_ok]
print(f"直立区間: {T_ok*dt:.2f} s / 全 {len(zc_)*dt:.2f} s")
fem = Q[:, :6]; T = len(fem); t = np.arange(T) * dt
if T * dt < 2.5: print("直立区間が短すぎる (<2.5s) — このランは不採用"); sys.exit(0)


def bandpass_hilbert(x, lo=3.0, hi=15.0):
    X = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(len(x), dt)
    X[(f < lo) | (f > hi)] = 0
    xa = np.fft.irfft(X, n=len(x))
    # 解析信号 (負周波数を落として2倍)
    Xa = np.fft.fft(xa); n = len(xa); h = np.zeros(n); h[0] = 1; h[1:n // 2] = 2; h[n // 2] = 1
    return np.fft.ifft(Xa * h)


phi = np.array([np.unwrap(np.angle(bandpass_hilbert(fem[:, i]))) for i in range(6)]).T   # (T, 6)
freq = np.diff(phi, axis=0).mean(0) / (2 * np.pi * dt)
print("脚ごとの位相振動数 [Hz]:", dict(zip(LEGS, np.round(freq, 2))))
sw = (C < 1e-8).mean(0)
print("遊脚率:", dict(zip(LEGS, np.round(sw, 2))))
# 複素コヒーレンス行列
E = np.exp(1j * phi)
Zc = (E.conj().T @ E) / T           # Z_ij = ⟨e^{i(φ_j−φ_i)}⟩ の共役; エルミート
print("\n脚間ロック強度 |Z_ij| (1=完全ロック):")
print("      " + " ".join(f"{l:>5s}" for l in LEGS))
for i, l in enumerate(LEGS):
    print(f"{l:>5s} " + " ".join(f"{abs(Zc[i, j]):5.2f}" for j in range(6)))
w, V = np.linalg.eigh(Zc)
v = V[:, -1]; ang = np.angle(v * np.conj(v[0]))
print(f"\n主固有値 {w[-1]:.2f} / 6 (集団同期の強さ)。主固有ベクトルの相対位相 [deg]:", dict(zip(LEGS, np.round(np.degrees(ang)))))
trip = {"LF": 0, "RM": 0, "LH": 0, "RF": 180, "LM": 180, "RH": 180}
dev = [abs(((np.degrees(ang[i]) - trip[l] + 180) % 360) - 180) for i, l in enumerate(LEGS)]
print("  三脚歩容 (LF,RM,LH 同相 / RF,LM,RH 逆相) からのずれ [deg]:", dict(zip(LEGS, np.round(dev))))
# Kuramoto フィット
dphi = np.gradient(phi, dt, axis=0)
rows = []; names = []
for i in range(6):
    X = [np.ones(T)]
    nm = ["w"]
    for j in range(6):
        if j == i: continue
        X.append(np.sin(phi[:, j] - phi[:, i])); X.append(np.cos(phi[:, j] - phi[:, i]))
        nm += [f"a_{LEGS[j]}", f"b_{LEGS[j]}"]
    X = np.array(X).T
    beta, *_ = np.linalg.lstsq(X, dphi[:, i], rcond=None)
    r2 = 1 - ((dphi[:, i] - X @ beta) ** 2).sum() / ((dphi[:, i] - dphi[:, i].mean()) ** 2).sum()
    rows.append(beta); names.append(nm)
    print(f"  {LEGS[i]}: 固有振動数 {beta[0]/(2*np.pi):.2f} Hz, R²={r2:.2f}, 最強結合 " +
          ", ".join(f"{nm[k]}={beta[k]:+.1f}" for k in np.argsort(-np.abs(beta[1:]))[:2] + 1))
# ロック状態まわりのヤコビアン: J_ik = ∂(dφ_i/dt)/∂φ_k
phi_lock = ang  # 主固有ベクトルの位相を平衡とみなす
J = np.zeros((6, 6))
for i in range(6):
    beta = rows[i]; k = 1
    for j in range(6):
        if j == i: continue
        a, b = beta[k], beta[k + 1]; k += 2
        d = phi_lock[j] - phi_lock[i]
        g = a * np.cos(d) - b * np.sin(d)          # ∂/∂φ_j of [a sin d + b cos d]
        J[i, j] += g; J[i, i] -= g
ev = np.linalg.eigvals(J)
print("\nロック状態のヤコビアン固有値 (実部<0 なら安定; 1つは並進対称で 0):")
print("  ", ", ".join(f"{e.real:+.1f}{e.imag:+.1f}i" for e in sorted(ev, key=lambda z: -z.real)))
json.dump(dict(freq=freq.tolist(), Zabs=np.abs(Zc).tolist(), Zang=np.angle(Zc).tolist(), lead_eig=float(w[-1]),
               lead_phase_deg=np.degrees(ang).tolist(), tripod_dev_deg=[float(x) for x in dev],
               jac_eig=[[float(e.real), float(e.imag)] for e in ev]),
          open("phase_algebra/outputs/walk_phase.json", "w"), indent=1)
print("保存: phase_algebra/outputs/walk_phase.json")
