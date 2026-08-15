#!/usr/bin/env python
"""閉ループ v2: 自己受容感覚 + 荷重感覚(campaniform sensilla相当)。

closed_loop.py への追加:
  1. 脚ごとの接地力 |F| → 感覚率に加算 (KL項)。接地/遊脚の状態が回路に入る。
     荷重は脚間協調(Cruseルール)の鍵とされる信号。
  2. Coxa(前後振り)ゲイン強化 0.5→0.9 — 振動を推進へ。
  3. 2.0秒実行で協調の発達を観察。
"""
import numpy as np
import pandas as pd
from brian2 import ms, Hz, prefs

import vnc_model
from integrate import MUSCLE2JOINT, SEG2LEG, SIDE2LEG, build_pools, DT_WIN, PHYS_DT, TAU_ACT, R0
from closed_loop import sensory_pools, NERVE2LEG

prefs.codegen.target = "numpy"

T_TOTAL = 2.0
MDN_RATE = 300
R_TONIC, KP, KV, R_MAX = 15.0, 50.0, 0.5, 150.0
KL = 50.0                       # 荷重感覚ゲイン [Hz]
GAIN = {"Coxa": 0.9, "Femur": 0.7, "Tibia": 0.8, "Tarsus1": 0.4}
Q_CLAMP, SLEW, ACT_MAX = 0.8, 0.06, 2.5
LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]


def main():
    from flygym import Fly, Camera, SingleFlySimulation

    sn_by_leg = sensory_pools()
    sn_all = [b for leg in sn_by_leg.values() for b in leg]
    net, mon, ids, idx, pg = vnc_model.make_network(
        vnc_model.MDN_BODYIDS, r_stim_hz=MDN_RATE, sensory_bodyids=sn_all)
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
    sense_j = {leg: [k for k, n in enumerate(joint_names)
                     if (f"{leg}Femur" in n or f"{leg}Tibia" in n)
                     and "roll" not in n] for leg in LEGS}
    # 接触センサ → 脚
    contact_names = [str(s) for s in fly.contact_sensor_placements]
    leg_contact = {leg: [k for k, n in enumerate(contact_names) if leg in n]
                   for leg in LEGS}

    n_win = int(T_TOTAL / DT_WIN)
    act = {k: 0.0 for k in pools}
    q_cmd_prev = q_neutral.copy()
    prev_count = np.zeros(len(ids))
    rates_vec = np.zeros(len(kept))
    f_ref = 1e-6
    log_act, log_pos, log_load = [], [], []

    for wi in range(n_win):
        qs, dqs = obs["joints"][0], obs["joints"][1]
        cf = obs["contact_forces"]
        loads = {}
        for leg in LEGS:
            F = float(np.linalg.norm(cf[leg_contact[leg]], axis=1).sum()) if len(leg_contact[leg]) else 0.0
            loads[leg] = F
        f_ref = max(f_ref, max(loads.values()))
        for leg in LEGS:
            js = sense_j[leg]
            dev = np.abs(qs[js] - q_neutral[js]).sum()
            vel = np.abs(dqs[js]).sum()
            r = R_TONIC + KP * dev + KV * vel + KL * (loads[leg] / f_ref)
            rates_vec[leg_slices[leg]] = np.clip(r, 0, R_MAX)
        pg.rates = rates_vec * Hz
        net.run(2 * ms)
        count = mon.count[:]
        d_spk = count - prev_count
        prev_count = count.copy()
        q = q_neutral.copy()
        for key, members in pools.items():
            rate = d_spk[members].sum() / len(members) / DT_WIN
            act[key] += DT_WIN * ((rate / R0) - act[key]) / TAU_ACT
            act[key] = min(max(act[key], 0.0), ACT_MAX)
        for (jname, d), k in jidx.items():
            base = jname.split("joint_")[1][2:]
            q[k] += GAIN[base] * d * act[(jname, d)]
        q = np.clip(q, q_neutral - Q_CLAMP, q_neutral + Q_CLAMP)
        q = np.clip(q, q_cmd_prev - SLEW, q_cmd_prev + SLEW)
        q_cmd_prev = q.copy()
        for _ in range(int(DT_WIN / PHYS_DT)):
            obs, *_ = sim.step({"joints": q})
        sim.render()
        log_act.append([act[k] for k in sorted(pools)])
        log_pos.append(obs["fly"][0].copy())
        log_load.append([loads[l] for l in LEGS])
        if wi % 200 == 0:
            print(f"  t={wi*DT_WIN:.2f}s spikes={int(count.sum())} "
                  f"load={np.round(list(loads.values()), 2)}")

    cam.save_video("outputs/closed_loop2.mp4")
    np.savez("outputs/closed_loop2_log.npz",
             act=np.array(log_act), act_cols=[f"{j}|{d}" for j, d in sorted(pools)],
             pos=np.array(log_pos), load=np.array(log_load))
    pos = np.array(log_pos)
    print(f"\ntotal spikes: {int(mon.count[:].sum())}")
    print(f"displacement: {np.linalg.norm(pos[-1,:2]-pos[0,:2]):.2f} mm "
          f"(x {pos[-1,0]-pos[0,0]:+.2f}, y {pos[-1,1]-pos[0,1]:+.2f})")
    print("saved outputs/closed_loop2.mp4")


if __name__ == "__main__":
    main()
