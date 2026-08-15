#!/usr/bin/env python
"""実測スパイク→力変換による歩行閉ループ (Azevedo et al. 2020, eLife 56754 準拠)。

従来 (closed_loop4): プール平均発火率 × 一定ゲイン → 「筋活動」(発明品)
本版: 運動ニューロン1本ごとに
  1. サイズ原理: プール内の入力シナプス数ランク(サイズ代理)で
     力/スパイクを 0.1〜10 μN に対数補間 (実測の100倍レンジ)
  2. 単収縮動態: α関数 (2段1次フィルタ)。fast側 τ=15ms (半値8.5ms),
     slow側 τ=150ms (500msでも飽和しない緩慢な蓄積) をサイズで補間
  3. 劣加算: tanh飽和 (2発で1.6倍、〜10発で飽和の実測に対応)
  4. 関節: Δq = k_cal × (ΣF_屈筋 − ΣF_伸筋)。k_cal だけが較正定数
"""
import numpy as np
import pandas as pd
from brian2 import ms, Hz, prefs

import vnc_model
from integrate import MUSCLE2JOINT, SEG2LEG, SIDE2LEG, DT_WIN, PHYS_DT
from closed_loop import sensory_pools
from closed_loop4 import compensation_factors, R_TONIC, KP, KV, KL, LEGS, MDN_RATE

prefs.codegen.target = "numpy"

T_TOTAL = 2.0
EXT_SCALE = 0.4
Q_CLAMP, SLEW = 0.6, 0.06
K_CAL = 0.012          # 較正定数 [rad/μN] (唯一の自由パラメータ)
GAIN_JOINT = {"Coxa": 1.1, "Femur": 0.9, "Tibia": 1.0, "Tarsus1": 0.5}


def build_measured_pools(idx, conns_path="../vnc-connectome/downloads/traced-connections.csv"):
    """MNごとの (ニューロンindex, 力ゲインμN, 時定数s, 所属関節・方向) を構築。"""
    mns = vnc_model.leg_mn_table()
    conns = pd.read_csv(conns_path)
    in_syn = conns.groupby("bodyId_post").weight.sum()
    rows = []
    for _, r in mns.iterrows():
        if r.target not in MUSCLE2JOINT or r.bodyid not in idx:
            continue
        joint, d = MUSCLE2JOINT[r.target]
        leg = SIDE2LEG[r.soma_side] + SEG2LEG[r.soma_neuromere]
        rows.append(dict(ni=idx[r.bodyid], joint=f"joint_{leg}{joint}", dir=d,
                         leg=leg, size=float(in_syn.get(r.bodyid, 1.0))))
    df = pd.DataFrame(rows)
    # サイズ原理: プール内ランク百分位 → 力/スパイク 0.1〜10μN (対数), τ 150→15ms
    df["pct"] = df.groupby(["joint", "dir"])["size"].rank(pct=True)
    df["gain_uN"] = 10.0 ** (-1.0 + 2.0 * df.pct)          # 0.1 .. 10 μN
    df["tau"] = 0.150 * (0.1) ** df.pct                     # 150ms .. 15ms
    df["fmax"] = 6.0 * df.gain_uN                           # 〜10発で飽和
    return df


def main():
    from flygym import Fly, Camera, SingleFlySimulation

    sn_by_leg = sensory_pools()
    sens_comp, motor_comp = compensation_factors(sn_by_leg)
    sn_all = [b for leg in sn_by_leg.values() for b in leg]
    net, mon, ids, idx, pg = vnc_model.make_network(
        vnc_model.MDN_BODYIDS, r_stim_hz=MDN_RATE, sensory_bodyids=sn_all)
    kept = [b for b in sn_all if b in idx]
    pos_of = {b: i for i, b in enumerate(kept)}
    leg_slices = {leg: np.array([pos_of[b] for b in bs if b in pos_of])
                  for leg, bs in sn_by_leg.items()}

    mp = build_measured_pools(idx)
    print(f"measured MNs: {len(mp)}  gain range [{mp.gain_uN.min():.2f}, "
          f"{mp.gain_uN.max():.1f}] μN/spike")
    ni = mp.ni.values
    gain = mp.gain_uN.values
    tau = mp.tau.values
    fmax = mp.fmax.values
    # α関数 = 2段1次フィルタ (u: 興奮, f: 力)
    u_state = np.zeros(len(mp))
    f_state = np.zeros(len(mp))
    joints = sorted(mp.joint.unique())
    jmask = {}   # (joint, dir) -> mask

    fly = Fly(init_pose="tripod", control="position", enable_adhesion=False)
    cam = Camera(fly=fly, play_speed=0.1)
    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=PHYS_DT)
    obs, _ = sim.reset()
    q_neutral = obs["joints"][0].copy()
    joint_names = list(fly.actuated_joints)
    jidx = {}
    for j in joints:
        m2 = [k for k, n in enumerate(joint_names) if j in n]
        if m2:
            jidx[j] = m2[0]
            for dd in (+1, -1):
                jmask[(j, dd)] = ((mp.joint == j) & (mp.dir == dd)).values
    sense_j = {leg: [k for k, n in enumerate(joint_names)
                     if (f"{leg}Femur" in n or f"{leg}Tibia" in n)
                     and "roll" not in n] for leg in LEGS}
    contact_names = [str(s) for s in fly.contact_sensor_placements]
    leg_contact = {leg: [k for k, n in enumerate(contact_names) if leg in n]
                   for leg in LEGS}

    n_win = int(T_TOTAL / DT_WIN)
    q_cmd_prev = q_neutral.copy()
    prev_count = np.zeros(len(ids))
    rates_vec = np.zeros(len(kept))
    f_ref = 1e-6
    log_force, log_pos, log_load, log_z = [], [], [], []

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
            r = (R_TONIC + KP * dev + KV * vel + KL * (loads[leg] / f_ref)) \
                * sens_comp[leg]
            rates_vec[leg_slices[leg]] = np.clip(r, 0, 250)
        pg.rates = rates_vec * Hz
        net.run(2 * ms)
        count = mon.count[:]
        d_spk = (count - prev_count)[ni]          # このMNの窓内スパイク数
        prev_count = count.copy()
        # 単収縮 (α関数): du/dt=(spikes/dt - u)/τ, df/dt=(u-f)/τ
        u_state += DT_WIN * (d_spk / DT_WIN - u_state) / tau
        f_state += DT_WIN * (u_state - f_state) / tau
        # α核の正規化: 1スパイクのピーク力 = gain [μN] (h_peak=1/(τe))
        F_lin = gain * f_state * (tau * np.e)
        F_mn = fmax * np.tanh(F_lin / fmax)      # μN, 劣加算飽和
        q = q_neutral.copy()
        for j, k in jidx.items():
            leg = j.replace("joint_", "")[:2]
            base = j.split("joint_")[1][2:]
            F_flex = F_mn[jmask[(j, +1)]].sum() if (j, +1) in jmask else 0.0
            F_ext = F_mn[jmask[(j, -1)]].sum() if (j, -1) in jmask else 0.0
            dq = K_CAL * GAIN_JOINT[base] * motor_comp[leg] * \
                (F_flex - EXT_SCALE * F_ext)
            q[k] += dq
        q = np.clip(q, q_neutral - Q_CLAMP, q_neutral + Q_CLAMP)
        q = np.clip(q, q_cmd_prev - SLEW, q_cmd_prev + SLEW)
        q_cmd_prev = q.copy()
        for _ in range(int(DT_WIN / PHYS_DT)):
            obs, *_ = sim.step({"joints": q})
        sim.render()
        log_force.append(F_mn.copy())
        log_pos.append(obs["fly"][0].copy())
        log_load.append([loads[l] for l in LEGS])
        log_z.append(obs["fly"][0][2])
        if wi % 200 == 0:
            stance = sum(1 for l in LEGS if loads[l] > 0.05 * f_ref)
            print(f"  t={wi*DT_WIN:.2f}s spikes={int(count.sum())} "
                  f"z={obs['fly'][0][2]:.2f} stance={stance} "
                  f"Fmax={F_mn.max():.1f}uN")

    cam.save_video("outputs/measured_interface.mp4")
    from brian2 import ms as _ms
    np.savez("outputs/measured_log.npz",
             force=np.array(log_force), gain=gain, tau=tau,
             joint=mp.joint.values.astype(str), dir=mp.dir.values,
             pos=np.array(log_pos), load=np.array(log_load), z=np.array(log_z),
             spikes_i=np.array(mon.i[:]), spikes_t=np.array(mon.t[:] / _ms) / 1000.0,
             ni=ni)
    pos = np.array(log_pos)
    load = np.array(log_load)
    thr = 0.05 * load.max()
    steps = {LEGS[k]: int((np.diff((load[:, k] > thr).astype(int)) == 1).sum())
             for k in range(6)}
    print(f"\nbody z mean {np.mean(log_z):.2f}  displacement "
          f"{np.linalg.norm(pos[-1,:2]-pos[0,:2]):.2f} mm")
    print("steps:", steps)
    print("saved outputs/measured_interface.mp4")


if __name__ == "__main__":
    main()
