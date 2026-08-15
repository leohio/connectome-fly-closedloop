#!/usr/bin/env python
"""統合22 Loop3-4: 視覚(オセリ)→FlyWire脳→DN→ヒンジの閉ループ飛行。

チェーン (全段コネクトーム由来):
  体傾き → オセリ視細胞273 (水平線符号化, レート2ms窓)
  → FlyWire視覚サブ回路470 LIF → DN発火率 (τ=50ms 低域)
  → DNg04左右差=ロール→クラッチ反対称 gd
    DNp18対称=ピッチ→回転タイミング dr  (統合21のヒンジ受け口)
ESポリシーは高度(z, vz)のみ。姿勢項は完全にゼロ = 姿勢制御は視覚反射だけ。

mode=smoke : ゲイン符号スイープ (短時間)
mode=main  : n=8 対照つき (OFF / VIS-REAL / VIS-SHUF / ES-ATT参照)
"""
import sys
import numpy as np
import mujoco
from brian2 import SpikeMonitor, Network, Hz, ms, prefs

import phase_reflex as PR
import brain_visual as BV

prefs.codegen.target = "numpy"

DT_W = 0.002          # 視覚ループの窓 (ドリフトは遅い)
TAU_DN = 0.02         # DNレートの低域 [s]
T_CAP = 3.0


def build_vision(shuffle=None, seed=0):
    neu, syn, ids, idx, meta = BV.build_subcircuit(shuffle=shuffle,
                                                   shuffle_seed=seed)
    pg, sh, ssign, is_oc = BV.sensory_attach(neu, idx, meta)
    mon = SpikeMonitor(neu, record=False)
    net = Network(neu, syn, mon, pg, sh)
    dn = meta["dn"].reset_index(drop=True)
    side = [str(meta["side"][idx[int(r.root_id)]]) for _, r in dn.iterrows()]
    gi = lambda ty, sd: [idx[int(r.root_id)] for k, (_, r) in
                         enumerate(dn.iterrows())
                         if r.primary_type == ty and side[k] == sd]
    groups = dict(rollL=gi("DNg04", "left"), rollR=gi("DNg04", "right"),
                  pitch=gi("DNp18", "left") + gi("DNp18", "right"))
    return net, mon, pg, ssign, is_oc, groups, len(ids)


def trial(G_roll, G_pitch, vision=True, shuffle=None, seed=0,
          es_att=False, es_full=False, es_rate=False, T=T_CAP):
    """(生存s, 直立度mean) を返す。es_att=True でESの姿勢項を復活(参照上限)"""
    from brian2 import ms as _ms
    env = PR._fly_env()
    K_full = env["K_att"]            # すでにω列ゼロ、e_b列あり
    b_pol, U_SCALE = env["b_pol"], env["U_SCALE"]
    Pw, Q0, ZT_W, m = env["Pw"], env["Q0"], env["ZT_W"], env["m"]
    K = K_full.copy()
    if es_full:                      # 完全ES (姿勢+ω, 統合11の上限参照)
        ckp = np.load("../fly-flight-sim/outputs/hover_policy.npz")
        K = ckp["theta"][:5*7].reshape(5, 7).copy()
        es_att = True
    elif es_rate:                    # ωダンピングのみ (姿勢項ゼロ)
        ckp = np.load("../fly-flight-sim/outputs/hover_policy.npz")
        K = ckp["theta"][:5*7].reshape(5, 7).copy()
        K[:, 2:4] = 0.0
        es_att = True                # u1-u4を有効化 (ω項の出力経路)
    elif not es_att:
        K[:, 2:4] = 0.0              # 姿勢項を除去 → 高度のみ
    if vision:
        net, mon, pg, ssign, is_oc, groups, nn = build_vision(shuffle, seed)
        # 較正: 水平で0.3s → DN基準レート
        f = {k: 0.0 for k in groups}
        prev = {k: 0.0 for k in groups}
        cnt0 = np.zeros(nn)
        pg.rates = BV.vis_rates(0, 0, 0, 0, ssign, is_oc) * Hz
        net.run(300 * _ms)
        c = mon.count[:].copy()
        base = {k: sum(c[i] for i in groups[k]) / 0.3 for k in groups}
        prev_c = c
    d = mujoco.MjData(m)
    dtp = m.opt.timestep
    aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
           for nm in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                      "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]}
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    R = np.zeros(9)
    fdn = dict(base) if vision else {}
    gdL = gdR = dr = 0.0
    ups, n_alive = [], 0
    n_win = int(T / DT_W)
    kk = max(Pw["sharp"], 1e-3)
    for wi in range(n_win):
        t = wi * DT_W
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        zc = np.array([R[2], R[5], R[8]])
        e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
        if vision:
            om = d.qvel[3:6]
            pg.rates = BV.vis_rates(e_b[0], e_b[1], om[0], om[1],
                                    ssign, is_oc) * Hz
            net.run(DT_W * 1000 * _ms)
            c = mon.count[:].copy()
            for k in groups:
                r_now = sum(c[i] - prev_c[i] for i in groups[k]) / DT_W
                fdn[k] += DT_W * (r_now - fdn[k]) / TAU_DN
            prev_c = c
            sig_roll = ((fdn["rollL"] - fdn["rollR"])
                        - (base["rollL"] - base["rollR"])) \
                / (base["rollL"] + base["rollR"] + 5.0)
            sig_pitch = (fdn["pitch"] - base["pitch"]) / (base["pitch"] + 5.0)
            gdL = float(np.clip(G_roll * sig_roll, -0.4, 0.4))
            gdR = -gdL
            dr = float(np.clip(G_pitch * sig_pitch, -0.4, 0.4))
        for kp in range(int(DT_W / dtp)):
            tt = t + kp * dtp
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zc = np.array([R[2], R[5], R[8]])
            eb2 = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
            omp = d.qvel[3:6]
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -d.qvel[2] / 30.0,
                          eb2[0], eb2[1],
                          omp[0] / 20.0, omp[1] / 20.0, omp[2] / 20.0])
            u = np.tanh(K @ x + b_pol) * U_SCALE
            amp = np.clip(1.0 + u[0], 0.5, 1.6)
            env0 = min(tt / 0.03, 1.0)
            ph2 = 2 * np.pi * Pw["freq"] * tt
            s = np.sin(ph2)
            rotL = np.tanh(kk * np.cos(ph2 + Pw["phase"] + dr)) / np.tanh(kk)
            rotR = rotL
            down = 1.0 if np.cos(ph2) < 0 else 0.0
            eL = env0 * amp * (1.0 + gdL * down)
            eR = env0 * amp * (1.0 + gdR * down)
            u1 = u[1] if es_att else 0.0
            u2 = u[2] if es_att else 0.0
            u3 = u[3] if es_att else 0.0
            u4 = u[4] if es_att else 0.0
            d.ctrl[:] = 0
            d.ctrl[aid["wing_yaw_left"]] = eL * (Pw["yaw_amp"] * s + u1 + u2)
            d.ctrl[aid["wing_yaw_right"]] = eR * (Pw["yaw_amp"] * s + u1 - u2)
            d.ctrl[aid["wing_pitch_left"]] = eL * (-Pw["pitch_amp"] * rotL
                                                   + Pw["pitch_bias"] + u3 + u4)
            d.ctrl[aid["wing_pitch_right"]] = eR * (-Pw["pitch_amp"] * rotR
                                                    + Pw["pitch_bias"] + u3 - u4)
            d.ctrl[aid["wing_roll_left"]] = eL * Pw["roll_amp"] * np.sin(2 * ph2)
            d.ctrl[aid["wing_roll_right"]] = eR * Pw["roll_amp"] * np.sin(2 * ph2)
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        n_alive = wi + 1
    return n_alive * DT_W, float(np.mean(ups)) if ups else 0.0


def mode_smoke():
    print("OFF (高度ESのみ):", flush=True)
    s, up = trial(0, 0, vision=False, T=1.0)
    print(f"  生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    for gr, gp in [(1.0, 1.0), (-1.0, -1.0), (2.0, -2.0), (-2.0, 2.0)]:
        s, up = trial(gr, gp, T=1.0)
        print(f"VIS G_roll={gr} G_pitch={gp}: 生存{s:.2f}s 直立度{up:+.2f}",
              flush=True)


def mode_main():
    NREP = 8
    Gr, Gp = (eval(sys.argv[2]) if len(sys.argv) > 2 else (4.0, -4.0))
    print(f"=== 統合22本実験 (n={NREP}, G_roll={Gr}, G_pitch={Gp}) ===",
          flush=True)
    s, up = trial(0, 0, vision=False)
    print(f"{'OFF':10s} 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    s, up = trial(0, 0, vision=False, es_att=True)
    print(f"{'ES-ATT参照':10s} 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    s, up = trial(0, 0, vision=False, es_full=True)
    print(f"{'ES-FULL上限':10s} 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    rows = []
    for name, sh, seed in [("VIS-REAL", None, 0), ("VIS-SHUF s0", "all", 0),
                           ("VIS-SHUF s1", "all", 1)]:
        ss, uu = [], []
        for rep in range(NREP):
            s, up = trial(Gr, Gp, shuffle=sh, seed=seed)
            ss.append(s); uu.append(up)
            print(f"  {name} rep{rep}: 生存{s:.2f}s 直立度{up:+.2f}",
                  flush=True)
        print(f"{name:10s} 生存{np.mean(ss):.2f}±{np.std(ss):.2f}s "
              f"直立度{np.mean(uu):+.2f}±SEM{np.std(uu)/np.sqrt(NREP):.2f}",
              flush=True)
        rows.append((name, ss, uu))
    np.savez("outputs/vision_main.npz", rows=np.array(rows, dtype=object))


def mode_combo():
    """最終実験: 共通のωダンピング基盤 + 視覚ドリフト補正。
    生物の分業 (ハルテア=速いω, 視覚=遅い姿勢) の機能的再現。"""
    NREP = 4
    Gr, Gp = 4.0, -4.0
    print("=== 統合22combo: ωダンピング基盤+視覚 (T=3s) ===", flush=True)
    s, up = trial(0, 0, vision=False, es_rate=True)
    print(f"{'RATEのみ':12s} 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    rows = [("RATE", [s], [up])]
    for name, sh, seed in [("VIS-REAL", None, 0), ("VIS-SHUF s0", "all", 0),
                           ("VIS-SHUF s1", "all", 1)]:
        ss, uu = [], []
        for rep in range(NREP):
            s, up = trial(Gr, Gp, shuffle=sh, seed=seed, es_rate=True)
            ss.append(s); uu.append(up)
            print(f"  {name}+RATE rep{rep}: 生存{s:.2f}s 直立度{up:+.2f}",
                  flush=True)
        print(f"{name+'+RATE':12s} 生存{np.mean(ss):.2f}±{np.std(ss):.2f}s "
              f"直立度{np.mean(uu):+.3f}±SEM{np.std(uu)/np.sqrt(NREP):.3f}",
              flush=True)
        rows.append((name, ss, uu))
    np.savez("outputs/vision_combo.npz", rows=np.array(rows, dtype=object))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    {"smoke": mode_smoke, "main": mode_main, "combo": mode_combo}[mode]()
