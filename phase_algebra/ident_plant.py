#!/usr/bin/env python
"""P1: 機体のストロボ写像 F, G の同定 (羽ばたき平均状態の ARX 版)。

初版 (位相0の瞬時状態で中心差分) は、瞬時角速度が羽ばたき振動 (±40 rad/s) と
翅関節の隠れ状態に支配され、9次元では写像がマルコフにならず |λ|=2.6 の偽固有値を
出した (線形予測が発散、実軌道は有界)。戦略第1層の通り、状態は**1羽ばたき平均の
遅い成分** (ハルテアが見る量) で定義し直す。

    x̄[k] = 羽ばたき k の間の平均 [e_b0, e_b1, ωx, ωy, ωz, z, vz, vx, vy]
    u[k] = 羽ばたき k の間に保持した翅変調 (5)

    x̄[k+1] = F x̄[k] + G1 u[k] + G0 u[k+1] + c

(G0 は同一羽ばたき内の直達項)。開ループは不安定なので短い軌道 (20羽ばたき) を
多数生成し、線形域 (|δe_b|<0.3, |δω|<30) のサンプルだけで最小二乗する。
参照は閉ループ (ES帰還則 + 真の遅いω) で1秒静定した状態。
"""
import os, sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import openloop_hover as OH

TH = np.load("outputs/bioflight_best.npy")
N_X, N_U = 7, 5
K_POL = TH[12:12 + N_U * N_X].reshape(N_U, N_X)
B_POL = TH[12 + N_U * N_X:]
P, U_TRIM = OH.unpack(TH[:12])
E = OH.env()
m, Q0, ZT_W, aid = E["m"], np.asarray(E["Q0"], float), E["ZT_W"], E["aid"]
PER = 1.0 / P["freq"]
STEPS = 100
m.opt.timestep = PER / STEPS
DTP = m.opt.timestep
TAU_TW = 0.00425
KK = max(P["sharp"], 1e-3)
Z_T = 12.0
NAMES = ["e_b0", "e_b1", "wx", "wy", "wz", "z", "vz", "vx", "vy"]
UNAMES = ["amp", "yaw_c", "yaw_d", "pitch_c", "pitch_d"]
LIN_EB, LIN_W = 0.3, 30.0


def wing_ctrl(d, u, t):
    amp = np.clip(1.0 + u[0], 0.5, 1.6)
    ph2 = 2 * np.pi * P["freq"] * t
    s = np.sin(ph2)
    rot = np.tanh(KK * np.cos(ph2 + P["phase"])) / np.tanh(KK)
    e = amp
    d.ctrl[:] = 0
    d.ctrl[aid["wing_yaw_left"]] = e * (P["yaw_amp"] * s + u[1] + u[2])
    d.ctrl[aid["wing_yaw_right"]] = e * (P["yaw_amp"] * s + u[1] - u[2])
    d.ctrl[aid["wing_pitch_left"]] = e * (-P["pitch_amp"] * rot + P["pitch_bias"] + u[3] + u[4])
    d.ctrl[aid["wing_pitch_right"]] = e * (-P["pitch_amp"] * rot + P["pitch_bias"] + u[3] - u[4])
    d.ctrl[aid["wing_roll_left"]] = e * P["roll_amp"] * np.sin(2 * ph2)
    d.ctrl[aid["wing_roll_right"]] = e * P["roll_amp"] * np.sin(2 * ph2)


def state_of(d):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    zc = np.array([R[2], R[5], R[8]])
    e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
    return np.array([e_b[0], e_b[1], *d.qvel[3:6], d.qpos[2], d.qvel[2], d.qvel[0], d.qvel[1]])


def settle(T=1.0):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = Z_T; d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    buf = np.zeros((STEPS, 3)); bi = 0; ssum = np.zeros(3)
    u_cmd = np.array(U_TRIM, float); u = np.array(U_TRIM, float)
    zlog = []
    for k in range(int(round(T / PER)) * STEPS):
        t = k * DTP
        ssum += d.qvel[3:6] - buf[bi]; buf[bi] = d.qvel[3:6].copy(); bi = (bi + 1) % STEPS
        if k % STEPS == 0:
            x = state_of(d); ow = ssum / STEPS
            xin = np.array([(Z_T - x[5]) / 5.0, -x[6] / 30.0, x[0], x[1], ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u_cmd = np.clip(U_TRIM + np.tanh(K_POL @ xin + B_POL) * 0.35, -0.55, 0.55)
            zlog.append(x[5])
        u += DTP * (u_cmd - u) / TAU_TW
        wing_ctrl(d, u, t)
        mujoco.mj_step(m, d)
    return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), time=d.time, u=u.copy()), np.array(zlog)


def restore(snap):
    d = mujoco.MjData(m)
    d.qpos[:] = snap["qpos"]; d.qvel[:] = snap["qvel"]
    if d.act.size: d.act[:] = snap["act"]
    d.time = snap["time"]; mujoco.mj_forward(m, d)
    return d


def perturb(d, dx):
    q = d.qpos[3:7].copy()
    for ax, ang in ((0, dx[0]), (1, dx[1])):
        if ang != 0.0:
            axis = np.zeros(3); axis[ax] = 1.0
            dq = np.zeros(4); mujoco.mju_axisAngle2Quat(dq, axis, ang)
            qn = np.zeros(4); mujoco.mju_mulQuat(qn, q, dq); q = qn
    d.qpos[3:7] = q
    d.qvel[3:6] += dx[2:5]; d.qpos[2] += dx[5]; d.qvel[2] += dx[6]; d.qvel[0] += dx[7]; d.qvel[1] += dx[8]
    mujoco.mj_forward(m, d)


def trajectory(snap, dx0, u_seq):
    """初期摂動 dx0、羽ばたきごとに保持する入力列 u_seq (L×5) で開ループ走行。
    各羽ばたきの平均状態 (L×9) を返す。"""
    d = restore(snap); perturb(d, dx0)
    t0 = d.time; xb = []
    for b, u in enumerate(u_seq):
        acc = np.zeros(9)
        for k in range(STEPS):
            wing_ctrl(d, u, t0 + (b * STEPS + k) * DTP)
            mujoco.mj_step(m, d)
            acc += state_of(d)
        xb.append(acc / STEPS)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
    return np.array(xb)


def main():
    print("静定中 (閉ループ 1.0s)...", flush=True)
    snap, zlog = settle(1.0)
    print(f"  高度 12.00 → {zlog[-1]:.2f}", flush=True)
    u_ref = snap["u"]
    # 参照の平均状態: u_ref 固定で3羽ばたき走らせた平均 (最初の羽ばたきを採用)
    xb_ref = trajectory(snap, np.zeros(9), np.tile(u_ref, (3, 1)))[0]
    print("  参照 (羽ばたき平均) x̄_ref:", np.round(xb_ref, 3))
    hx = np.array([0.05, 0.05, 3.0, 3.0, 3.0, 0.3, 2.0, 2.0, 2.0])
    hu = np.array([0.04, 0.04, 0.04, 0.04, 0.04])
    rg = np.random.default_rng(0)
    L, NTR = 20, 48
    rows_x, rows_u1, rows_u0, rows_y = [], [], [], []
    held = []
    for tr in range(NTR):
        dx0 = rg.normal(0, 0.4, 9) * hx
        du = rg.normal(0, 0.5, (L, 5)) * hu
        if tr % 6 == 0:                # 1/6 は入力ゼロ (自由応答)
            du[:] = 0
        xb = trajectory(snap, dx0, u_ref + du) - xb_ref
        if tr >= NTR - 6:              # 最後の6本は検証用に保持
            held.append((dx0, du, xb)); continue
        for k in range(len(xb) - 1):
            if np.abs(xb[k][:2]).max() > LIN_EB or np.abs(xb[k][2:5]).max() > LIN_W:
                break
            rows_x.append(xb[k]); rows_u1.append(du[k]); rows_u0.append(du[k + 1]); rows_y.append(xb[k + 1])
    X = np.array(rows_x); U1 = np.array(rows_u1); U0 = np.array(rows_u0); Y = np.array(rows_y)
    print(f"  学習サンプル: {len(X)} 羽ばたき (線形域内)", flush=True)
    Phi = np.hstack([X, U1, U0, np.ones((len(X), 1))])
    Theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    F = Theta[:9].T; G1 = Theta[9:14].T; G0 = Theta[14:19].T; c = Theta[19]
    resid = Y - Phi @ Theta
    r2 = 1 - (resid ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    print("\n1ステップ予測の R²:", dict(zip(NAMES, np.round(r2, 3))))
    ev, V = np.linalg.eig(F)
    order = np.argsort(-np.abs(ev))
    print("\nF の固有値 (|λ| 降順):")
    for i in order:
        lam = ev[i]; a = np.angle(lam)
        s = f"  |λ|={abs(lam):.4f} arg={np.degrees(a):+7.1f}°  "
        if abs(lam) > 1:
            dbl = np.log(2) / np.log(abs(lam)); s += f"不安定: 倍加 {dbl:.1f}羽ばたき ({dbl*PER*1000:.0f}ms)"
        else:
            s += "安定"
        if abs(a) > 1e-3 and abs(a) < np.pi - 1e-3:
            s += f"  振動周期 {2*np.pi/abs(a)*PER*1000:.0f}ms"
        w = np.abs(V[:, i]); top = np.argsort(-w)[:3]
        s += "   主成分: " + ", ".join(f"{NAMES[j]}({w[j]/w.max():.2f})" for j in top)
        print(s)
    print("\nG1 (状態 ← 前羽ばたきの入力):")
    for i, nm in enumerate(NAMES):
        print(f"  {nm:5s}: " + "  ".join(f"{UNAMES[j]}={G1[i, j]:+8.3f}" for j in range(5)))
    print("\n検証 (保持した6軌道、線形モデルの多段予測 vs シミュレーション):")
    for k_eval in (3, 6, 10, 15):
        errs = []
        for dx0, du, xb in held:
            if len(xb) <= k_eval: continue
            p = xb[0].copy()
            for k in range(k_eval):
                p = F @ p + G1 @ du[k] + G0 @ du[k + 1] + c
            errs.append(np.linalg.norm((xb[k_eval] - p)[[0, 1, 5, 6]]) / (np.linalg.norm(xb[k_eval][[0, 1, 5, 6]]) + 1e-9))
        print(f"  k={k_eval:2d}: 姿勢・高度の相対予測誤差 中央値={np.median(errs):.2f} (n={len(errs)})")
    np.savez("phase_algebra/outputs/plant_FG.npz", F=F, G1=G1, G0=G0, c=c, xb_ref=xb_ref, u_ref=u_ref,
             names=np.array(NAMES), unames=np.array(UNAMES), per=PER, r2=r2, eig=ev, n_samples=len(X))
    print("\n保存: phase_algebra/outputs/plant_FG.npz", flush=True)


if __name__ == "__main__":
    main()
