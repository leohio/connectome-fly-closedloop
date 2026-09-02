#!/usr/bin/env python
"""P2 検証: 閉ループを実際に走らせ、姿勢偏差の成長率 (1羽ばたきあたり) を遅延ごとに測る。

線形モデル (closed_loop.py) と同じ座標・同じ感覚模型 (τ=12ms の一次遅れ + ゲイン +
純遅延 d 羽ばたき、ノイズなし) で非線形シミュレーションを回し、
小さな姿勢摂動 (0.02 rad) からの |δe_b| の包絡線を指数フィットして |λ|_emp を得る。
予測 |λ|_pred (T の速いモードの最大絶対値) と突き合わせるのが目的。
"""
import os, sys, json
import numpy as np
import mujoco
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import phase_algebra.ident_plant as IP

PL = np.load("phase_algebra/outputs/plant_FG.npz")
XB_REF = PL["xb_ref"]
SF = json.load(open("outputs/sensor_fit.json"))
GS = np.array([SF[a]["G"] for a in "xyz"]); TAU_S = SF["x"]["tau_ms"] / 1000.0
A_S = np.exp(-IP.PER / TAU_S)


def run(K, b, d, T=3.0, pert=0.02, snap=None):
    d_ = IP.restore(snap)
    IP.perturb(d_, np.array([pert, pert, 0, 0, 0, 0, 0, 0, 0]))
    u_cmd = snap["u"].copy(); u = snap["u"].copy()
    om_f = GS * XB_REF[2:5]; hist = [om_f.copy() for _ in range(d + 1)]
    t0 = d_.time; xb_log = []; acc = np.zeros(9); buf = np.zeros((IP.STEPS, 3)); bi = 0; ssum = np.zeros(3)
    n_beats = int(T / IP.PER)
    for k in range(n_beats * IP.STEPS):
        t = t0 + k * IP.DTP
        ssum += d_.qvel[3:6] - buf[bi]; buf[bi] = d_.qvel[3:6].copy(); bi = (bi + 1) % IP.STEPS
        if k % IP.STEPS == 0 and k > 0:
            xb_log.append(acc / IP.STEPS); acc[:] = 0
            om_bar = ssum / IP.STEPS
            om_f = A_S * om_f + (1 - A_S) * GS * om_bar          # 回路の一次遅れ
            hist.insert(0, om_f.copy()); hist = hist[:d + 1]
            om_hat = hist[d]
            x = IP.state_of(d_)
            xin = np.array([(IP.Z_T - x[5]) / 5.0, -x[6] / 30.0, x[0], x[1], om_hat[0] / 20, om_hat[1] / 20, om_hat[2] / 20])
            u_cmd = np.clip(IP.U_TRIM + np.tanh(K @ xin + b) * 0.35, -0.55, 0.55)
        u += IP.DTP * (u_cmd - u) / IP.TAU_TW
        IP.wing_ctrl(d_, u, t)
        mujoco.mj_step(m := IP.m, d_)
        acc += IP.state_of(d_)
        if not np.isfinite(d_.qpos[2]) or d_.qpos[2] < 0.5 or d_.qpos[2] > 40:
            break
    xb = np.array(xb_log) - XB_REF
    return xb, (k + 1) * IP.DTP


def growth_rate(xb, per_beats=30):
    """|δe_b| の包絡 (per_beats 窓の最大) の対数を、線形域 (<0.5 rad) で直線フィット"""
    a = np.linalg.norm(xb[:, :2], axis=1)
    env = np.array([a[max(0, i - per_beats):i + 1].max() for i in range(len(a))])
    ok = np.where(env < 0.5)[0]
    if len(ok) < 40: return np.nan, len(ok)
    i0, i1 = per_beats, ok[-1]
    if i1 - i0 < 30: return np.nan, len(ok)
    sl = np.polyfit(np.arange(i0, i1), np.log(env[i0:i1] + 1e-9), 1)[0]
    return float(np.exp(sl)), int(i1)


if __name__ == "__main__":
    print("静定中...", flush=True)
    snap, _ = IP.settle(1.0)
    ctrls = [("ES-K (bioflight_best)", "outputs/bioflight_best.npy"),
             ("報酬学習 遅延2訓練 個体2 (v6)", "outputs/reward_K_v6_delay2.json:2")]
    from phase_algebra.closed_loop import load_K, build_T
    res = {}
    for name, src in ctrls:
        K, b = load_K(src)
        print(f"\n{name}")
        print("  d   生存s   |λ|_emp(実測成長率)   |λ|_pred(速いモード)   |λ|_pred(全体)")
        rows = []
        for d in range(0, 5):
            xb, srv = run(K, b, d, T=4.0, snap=snap)
            lam_e, n_lin = growth_rate(xb)
            ev = np.linalg.eigvals(build_T(K, b, d, False, "circuit"))
            fast = [l for l in ev if 1e-3 < abs(np.angle(l)) < np.pi - 1e-3 and 2 * np.pi / abs(np.angle(l)) * IP.PER < 0.3]
            lam_fast = max(abs(l) for l in fast) if fast else np.nan
            lam_all = max(abs(ev))
            print(f"  {d}   {srv:5.2f}   {lam_e:8.4f} (線形域{n_lin}羽ばたき)     {lam_fast:8.4f}            {lam_all:8.4f}", flush=True)
            rows.append(dict(d=d, survival=srv, lam_emp=lam_e, lam_fast=lam_fast, lam_all=lam_all))
        res[name] = rows
    json.dump(res, open("phase_algebra/outputs/hover_growth.json", "w"), indent=1, ensure_ascii=False)
    print("\n保存: phase_algebra/outputs/hover_growth.json")
