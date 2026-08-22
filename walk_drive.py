#!/usr/bin/env python
"""統合38b: MDN刺激 → MANC腹髄net → 脚MNスパイク → flybody関節駆動。

統合1-7 (flygym) の界面をflybodyへ移植する第一歩。
- MN→筋: integrate_measured のサイズ原理プール (Azevedo 2020: 力/スパイク
  0.1-10μN 対数勾配、τ 150→15ms、飽和) をそのまま使う
- 関節対応: MANC (神経節T1-3, 側LHS/RHS, 標的筋) → flybodyの
  {coxa,femur,tibia,tarsus}_{T1..T3}_{left,right}
- 駆動: 屈筋力-伸筋力の差を立位姿勢まわりの角度指令に変換 (K_CAL, 統合16と同じ)

観察: MDN刺激下で (a) 立位が保たれるか (b) 関節が振動的に動くか (c) 体が動くか
"""
import sys
import numpy as np
import pandas as pd
import mujoco
from brian2 import Hz, ms as _ms, prefs
prefs.codegen.target = "numpy"
import vnc_model
import integrate_measured as IM
import walk_base as WB

DT_WIN = 0.002
K_CAL = 0.08   # flybody較正: サーボ応答80-95%実測、|F|~5μNでΔq~0.4radが出る値
GAIN_JOINT = {"coxa": 1.1, "femur": 0.9, "tibia": 1.0, "tarsus": 0.5}
SEG = {"F": "T1", "M": "T2", "H": "T3"}
SIDE = {"L": "left", "R": "right"}
J2FB = {"Coxa": "coxa", "Femur": "femur", "Tibia": "tibia",
        "Tarsus1": "tarsus"}


def flybody_pools(idx, W):
    """サイズ原理プールをflybodyアクチュエータへ対応づける"""
    df = IM.build_measured_pools()
    an = {n: i for i, n in enumerate(W["leg_names"])}
    pools = {}
    for _, r in df.iterrows():
        if r.bodyid not in idx:
            continue
        leg, joint = r.leg, r.joint.split("_")[1][2:]
        if joint not in J2FB:
            continue
        act = f"{J2FB[joint]}_{SEG[leg[1]]}_{SIDE[leg[0]]}"
        if act not in an:
            continue
        key = (act, int(r.dir))
        pools.setdefault(key, []).append(
            (idx[r.bodyid], float(r.gain_uN), float(r.tau), float(r.fmax)))
    return pools


TOTLOG = {}


def run(mdn_hz=150.0, T=2.0, video=None):
    W = WB.env()
    m = W["m"]
    an = {n: i for i, n in enumerate(W["leg_names"])}
    act_ids = {n: int(W["leg_act"][i]) for n, i in an.items()}
    stance = {n: float(W["stance_q"][i]) for n, i in an.items()}
    net, mon, ids, idx = vnc_model.make_network(vnc_model.MDN_BODYIDS,
                                                r_stim_hz=mdn_hz)
    pools = flybody_pools(idx, W)
    n_mn = sum(len(v) for v in pools.values())
    print(f"プール: {len(pools)}関節方向, MN {n_mn}本", flush=True)
    F = {}          # (act,dir) -> 筋力状態
    for k in pools:
        F[k] = np.zeros(len(pools[k]))
    d = mujoco.MjData(m)
    WB.reset_standing(m, d)
    WB.settle(m, d, W, T=0.3)
    x0 = d.qpos[0]
    prev = np.zeros(mon.source.N)
    renderer, frames = None, []
    if video:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = 1.2, -12, 100
    n_phys = int(DT_WIN / m.opt.timestep)
    qlog = []
    for wi in range(int(T / DT_WIN)):
        net.run(DT_WIN * 1000 * _ms)
        cnt = mon.count[:].copy()
        dspk = cnt - prev
        prev = cnt
        qcmd = dict(stance)
        for (act, dr), lst in pools.items():
            f = F[(act, dr)]
            for q, (ni, g, tau, fmax) in enumerate(lst):
                f[q] += dspk[ni] * g
                f[q] = min(f[q], fmax)
                f[q] -= DT_WIN * f[q] / tau
            tot = float(f.sum())
            TOTLOG.setdefault((act, dr), []).append(tot)
            qcmd[act] = qcmd.get(act, stance[act]) \
                + dr * K_CAL * GAIN_JOINT[act.split("_")[0]] * tot
        for n, i in act_ids.items():
            lo, hi = m.actuator_ctrlrange[i]
            d.ctrl[i] = np.clip(qcmd[n], lo, hi)
        for _ in range(n_phys):
            mujoco.mj_step(m, d)
        if renderer and wi % int(1 / 60 / DT_WIN + 0.5) == 0:
            cam.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
        qlog.append([d.qpos[m.jnt_qposadr[m.actuator_trnid[act_ids[n]][0]]]
                     for n in ["femur_T2_left", "tibia_T2_left"]])
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    q = np.array(qlog)
    print(f"MDN={mdn_hz:.0f}Hz {T:.1f}s: 直立度={R[8]:+.2f} z={d.qpos[2]:+.3f} "
          f"前進={d.qpos[0]-x0:+.4f} 側方={d.qpos[1]:+.4f}", flush=True)
    print(f"  中脚femur角: 平均{q[:,0].mean():+.3f} 振幅(std){q[:,0].std():.4f} "
          f"tibia: {q[:,1].mean():+.3f} / {q[:,1].std():.4f}", flush=True)
    stats = sorted(((np.mean(np.abs(v)), np.std(v), k)
                    for k, v in TOTLOG.items()), reverse=True)
    print("  筋力合計 |F| 上位:", flush=True)
    for mabs, sd, k in stats[:6]:
        print(f"    {k[0]:20s} dir{k[1]:+d}: 平均{mabs:7.2f}μN std{sd:6.2f}",
              flush=True)
    if video and frames:
        import imageio
        imageio.mimsave(video, frames, fps=60, quality=8)
        print("動画:", video, flush=True)


if __name__ == "__main__":
    hz = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    vid = sys.argv[3] if len(sys.argv) > 3 else None
    run(mdn_hz=hz, T=T, video=vid)
