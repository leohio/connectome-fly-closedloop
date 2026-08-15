#!/usr/bin/env python
"""統合21 Loop1: ヒンジ非線形の「受け口」検証 (脳なし・開ループ・決定論)。

問い: flybody の翅駆動で、以下の2つの位相依存メカニズムは
空力的な力・トルクを実際に生むか? (生まないなら界面を作っても無意味)
  1. 回転タイミング dr: 迎角切り返し位相のシフト
     rot = tanh(k·cos(ph + phase + dr))/tanh(k)
     (Dickinson robofly: 回転の前進/遅延で揚力が大きく変わる)
  2. クラッチ gd: 打ち下ろし半周期のみの伝達ゲイン
     e_eff = e·(1 + gd·D(ph)),  D=1 if cos(ph)<0 (打ち下ろし) else 0
     (b1張力→基礎骨片→半ストローク非対称の伝達、の機能モデル)

参照: 従来の線形振幅非対称 (ampL=1+a, ampR=1-a) との力権限比較。
各条件 T=0.12s、終端の角速度 ω と鉛直速度 vz を報告。
"""
import numpy as np
import mujoco
import phase_reflex as PR


def probe(drL=0.0, drR=0.0, gdL=0.0, gdR=0.0, aL=1.0, aR=1.0, T=0.12):
    env = PR._fly_env()
    Pw, Q0, m = env["Pw"], env["Q0"], env["m"]
    d = mujoco.MjData(m)
    dtp = m.opt.timestep
    aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
           for nm in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                      "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]}
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    kk = max(Pw["sharp"], 1e-3)
    n = int(T / dtp)
    for k in range(n):
        t = k * dtp
        env0 = min(t / 0.03, 1.0)
        ph2 = 2 * np.pi * Pw["freq"] * t
        s = np.sin(ph2)
        down = 1.0 if np.cos(ph2) < 0 else 0.0     # 打ち下ろし半周期
        rotL = np.tanh(kk * np.cos(ph2 + Pw["phase"] + drL)) / np.tanh(kk)
        rotR = np.tanh(kk * np.cos(ph2 + Pw["phase"] + drR)) / np.tanh(kk)
        eL = env0 * aL * (1.0 + gdL * down)
        eR = env0 * aR * (1.0 + gdR * down)
        d.ctrl[:] = 0
        d.ctrl[aid["wing_yaw_left"]] = eL * Pw["yaw_amp"] * s
        d.ctrl[aid["wing_yaw_right"]] = eR * Pw["yaw_amp"] * s
        d.ctrl[aid["wing_pitch_left"]] = eL * (-Pw["pitch_amp"] * rotL
                                               + Pw["pitch_bias"])
        d.ctrl[aid["wing_pitch_right"]] = eR * (-Pw["pitch_amp"] * rotR
                                                + Pw["pitch_bias"])
        d.ctrl[aid["wing_roll_left"]] = eL * Pw["roll_amp"] * np.sin(2 * ph2)
        d.ctrl[aid["wing_roll_right"]] = eR * Pw["roll_amp"] * np.sin(2 * ph2)
        mujoco.mj_step(m, d)
    return dict(vz=float(d.qvel[2]), wx=float(d.qvel[3]),
                wy=float(d.qvel[4]), wz=float(d.qvel[5]))


def show(label, r, base):
    print(f"{label:28s} vz={r['vz']:+7.2f} (Δ{r['vz']-base['vz']:+6.2f})  "
          f"ωx={r['wx']:+7.2f} ωy={r['wy']:+7.2f} ωz={r['wz']:+7.2f}",
          flush=True)


if __name__ == "__main__":
    base = probe()
    show("基準 (対称ホバー)", base, base)
    print("--- 回転タイミング dr (対称: 揚力/ピッチ) ---")
    for dr in [0.2, -0.2, 0.4, -0.4]:
        show(f"dr両翅 {dr:+.1f}", probe(drL=dr, drR=dr), base)
    print("--- 回転タイミング dr (反対称: ロール/ヨー) ---")
    for dr in [0.1, 0.2, 0.3]:
        show(f"drL=+{dr:.1f} drR=-{dr:.1f}", probe(drL=dr, drR=-dr), base)
    print("--- クラッチ gd (対称: 揚力) ---")
    for gd in [0.2, -0.2]:
        show(f"gd両翅 {gd:+.1f}", probe(gdL=gd, gdR=gd), base)
    print("--- クラッチ gd (反対称: ロール) ---")
    for gd in [0.1, 0.2]:
        show(f"gdL=+{gd:.1f} gdR=-{gd:.1f}", probe(gdL=gd, gdR=-gd), base)
    print("--- 参照: 線形振幅非対称 (統合18-20の界面) ---")
    for a in [0.1, 0.2]:
        show(f"ampL=1+{a:.1f} ampR=1-{a:.1f}", probe(aL=1+a, aR=1-a), base)
