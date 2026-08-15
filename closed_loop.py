#!/usr/bin/env python
"""閉ループ版: 自己受容感覚フィードバック付きの脳-身体連成。

integrate.py との違い:
  MuJoCo の関節角・角速度 → 脚別の自己受容感覚ニューロン(1,347本)の発火率
  というフィードバック経路を追加し、感覚→腹髄→運動→身体→感覚のループを閉じる。

感覚モデル(v1, 簡略):
  各脚の Femur/Tibia 関節について
  rate = clip(R_TONIC + KP*|Δq| + KV*|dq/dt|, 0, R_MAX)
  を、その脚の神経(ProLN/MesoLN/MetaLN × L/R)から入る proprioceptive SN 全員に与える。
"""
import numpy as np
import pandas as pd
from brian2 import ms, Hz, prefs

import vnc_model
from integrate import (MUSCLE2JOINT, GAIN, SEG2LEG, SIDE2LEG, build_pools,
                       DT_WIN, PHYS_DT, TAU_ACT, R0)

prefs.codegen.target = "numpy"

T_TOTAL = 1.0
MDN_RATE = 300
R_TONIC, KP, KV, R_MAX = 15.0, 50.0, 0.5, 120.0    # 感覚写像パラメータ
Q_CLAMP = 0.8      # 関節変位の上限 [rad]
SLEW = 0.06        # 1窓あたりの関節指令変化上限 [rad]
ACT_MAX = 2.5      # 筋活動の上限

NERVE2LEG = {"ProLN_L": "LF", "ProLN_R": "RF", "MesoLN_L": "LM",
             "MesoLN_R": "RM", "MetaLN_L": "LH", "MetaLN_R": "RH"}


def sensory_pools():
    props = pd.read_feather("neuron-properties.feather")
    sn = props[(props["class"] == "sensory neuron")
               & (props.modality == "proprioceptive")
               & props.entryNerve.isin(NERVE2LEG)]
    return {leg: sn[sn.entryNerve == nerve].bodyId.tolist()
            for nerve, leg in NERVE2LEG.items()}


def main():
    from flygym import Fly, Camera, SingleFlySimulation

    sn_by_leg = sensory_pools()
    sn_all = [b for leg in sn_by_leg.values() for b in leg]
    print("proprioceptive SNs:", {k: len(v) for k, v in sn_by_leg.items()})

    net, mon, ids, idx = None, None, None, None
    net, mon, ids, idx, pg = vnc_model.make_network(
        vnc_model.MDN_BODYIDS, r_stim_hz=MDN_RATE, sensory_bodyids=sn_all)
    # pg の並び = sn_all の並び(idx にあるもの)。脚→pgスライスを作る
    kept = [b for b in sn_all if b in idx]
    pos_of = {b: i for i, b in enumerate(kept)}
    leg_slices = {leg: np.array([pos_of[b] for b in bs if b in pos_of])
                  for leg, bs in sn_by_leg.items()}

    pools = build_pools(idx)
    fly = Fly(init_pose="stretch", control="position", enable_adhesion=False)
    cam = Camera(fly=fly, play_speed=0.1)
    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=PHYS_DT)
    obs, _ = sim.reset()
    q_neutral = obs["joints"][0].copy()
    joint_names = list(fly.actuated_joints)
    jidx = {}
    for (jname, d) in pools:
        m = [k for k, n in enumerate(joint_names) if jname in n]
        if m:
            jidx[(jname, d)] = m[0]
    # 感覚読み出し用: 脚→(Femur, Tibia)関節index
    legs = ["LF", "LM", "LH", "RF", "RM", "RH"]
    sense_j = {leg: [k for k, n in enumerate(joint_names)
                     if (f"{leg}Femur" in n or f"{leg}Tibia" in n)
                     and "roll" not in n] for leg in legs}

    n_win = int(T_TOTAL / DT_WIN)
    act = {k: 0.0 for k in pools}
    q_cmd_prev = q_neutral.copy()
    prev_count = np.zeros(len(ids))
    rates_vec = np.zeros(len(kept))
    log_act, log_pos, log_rate = [], [], []

    for wi in range(n_win):
        # --- 感覚: 関節状態 → SN発火率
        qs, dqs = obs["joints"][0], obs["joints"][1]
        for leg in legs:
            js = sense_j[leg]
            dev = np.abs(qs[js] - q_neutral[js]).sum()
            vel = np.abs(dqs[js]).sum()
            r = np.clip(R_TONIC + KP * dev + KV * vel, 0, R_MAX)
            rates_vec[leg_slices[leg]] = r
        pg.rates = rates_vec * Hz
        # --- 神経 2ms
        net.run(2 * ms)
        count = mon.count[:]
        d_spk = count - prev_count
        prev_count = count.copy()
        # --- 筋・関節
        q = q_neutral.copy()
        for key, members in pools.items():
            rate = d_spk[members].sum() / len(members) / DT_WIN
            act[key] += DT_WIN * ((rate / R0) - act[key]) / TAU_ACT
            act[key] = min(max(act[key], 0.0), ACT_MAX)
        for (jname, d), k in jidx.items():
            base = jname.split("joint_")[1][2:]
            q[k] += GAIN[base] * d * act[(jname, d)]
        # 飽和 + スルーレート制限(発散防止)
        q = np.clip(q, q_neutral - Q_CLAMP, q_neutral + Q_CLAMP)
        q = np.clip(q, q_cmd_prev - SLEW, q_cmd_prev + SLEW)
        q_cmd_prev = q.copy()
        # --- 物理 2ms
        for _ in range(int(DT_WIN / PHYS_DT)):
            obs, *_ = sim.step({"joints": q})
        sim.render()
        log_act.append([act[k] for k in sorted(pools)])
        log_pos.append(obs["fly"][0].copy())
        log_rate.append(rates_vec[[leg_slices[l][0] for l in legs if len(leg_slices[l])]].copy())
        if wi % 100 == 0:
            print(f"  t={wi*DT_WIN:.2f}s spikes={int(count.sum())} "
                  f"sens={np.round(log_rate[-1],0)}")

    cam.save_video("outputs/closed_loop.mp4")
    np.savez("outputs/closed_loop_log.npz",
             act=np.array(log_act), act_cols=[f"{j}|{d}" for j, d in sorted(pools)],
             pos=np.array(log_pos), sens=np.array(log_rate))
    pos = np.array(log_pos)
    print(f"\ntotal spikes: {int(mon.count[:].sum())}")
    print(f"displacement: {np.linalg.norm(pos[-1,:2]-pos[0,:2]):.2f} mm")
    print("saved outputs/closed_loop.mp4")


if __name__ == "__main__":
    main()
