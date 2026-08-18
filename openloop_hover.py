#!/usr/bin/env python
"""統合31: 帰還なしで安定にホバリングする翅運動学を探す。

統合30で、現行の飛行解は≥6.5kHzの作動更新を要し神経筋系では実装不可能と判明した。
根本原因は hover_params が方策と共最適化されており、基本波形単独ではホバー
できない (定数u で0.33s) こと。

そこで翅運動学パラメタ自体 + 静的トリムを、**帰還を一切使わずに**
長く安定に浮くよう最適化する。開ループ安定なホバーが得られれば、
その上に1羽ばたき1回のハルテア帰還 (コネクトーム回路が実際に出せる帯域) を
載せた、生物学的に実装可能な飛行制御が構成できる。

探索パラメタ (11): freq, yaw_amp, pitch_amp, pitch_bias, roll_amp, phase,
sharp, u0..u3 の静的トリム。帰還は無し (u は時間に依らず一定)。
"""
import json
import os
import sys
import numpy as np
import mujoco

CKPT = "outputs/openloop.json"
T_EVAL = 3.0
N_PERT = 3
N_PAIRS = 10
LR = 0.35
WORKERS = 12
KEYS = ["freq", "yaw_amp", "pitch_amp", "pitch_bias", "roll_amp",
        "phase", "sharp"]
_E = {}


def env():
    if _E:
        return _E
    import phase_reflex as PR
    import connectome_fastloop as CF
    e = PR._fly_env()
    m = CF.OPT.get_model()
    _E.update(Pw0=dict(e["Pw"]), Q0=e["Q0"], ZT_W=e["ZT_W"], m=m)
    _E["aid"] = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                 for n in ["wing_yaw_left", "wing_roll_left",
                           "wing_pitch_left", "wing_yaw_right",
                           "wing_roll_right", "wing_pitch_right"]}
    return _E


def unpack(theta):
    """theta は基準値に対する乗法/加法の摂動 (基準=hover_params)"""
    E = env()
    P = dict(E["Pw0"])
    P["freq"] = E["Pw0"]["freq"] * np.exp(0.15 * theta[0])
    P["yaw_amp"] = E["Pw0"]["yaw_amp"] * np.exp(0.3 * theta[1])
    P["pitch_amp"] = E["Pw0"]["pitch_amp"] * np.exp(0.3 * theta[2])
    P["pitch_bias"] = E["Pw0"]["pitch_bias"] + 0.3 * theta[3]
    P["roll_amp"] = E["Pw0"]["roll_amp"] * np.exp(0.3 * theta[4])
    P["phase"] = E["Pw0"]["phase"] + 0.5 * theta[5]
    P["sharp"] = max(E["Pw0"]["sharp"] * np.exp(0.3 * theta[6]), 1e-3)
    u = np.clip(theta[7:12] * 0.3, -0.55, 0.55)   # 静的トリム u0..u4
    return P, u


def rollout(theta, pert_seed=None, T=T_EVAL):
    E = env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    P, u = unpack(theta)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    if pert_seed is not None:
        d.qvel[3:6] = np.random.default_rng(pert_seed).normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
    dtp = m.opt.timestep
    kk = max(P["sharp"], 1e-3)
    R = np.zeros(9)
    ups, alive, zs = [], 0, []
    for k in range(int(T / dtp)):
        t = k * dtp
        amp = np.clip(1.0 + u[0], 0.5, 1.6)
        env0 = min(t / 0.03, 1.0)
        ph2 = 2 * np.pi * P["freq"] * t
        s = np.sin(ph2)
        rot = np.tanh(kk * np.cos(ph2 + P["phase"])) / np.tanh(kk)
        e = env0 * amp
        d.ctrl[:] = 0
        d.ctrl[aid["wing_yaw_left"]] = e * (P["yaw_amp"] * s + u[1] + u[2])
        d.ctrl[aid["wing_yaw_right"]] = e * (P["yaw_amp"] * s + u[1] - u[2])
        d.ctrl[aid["wing_pitch_left"]] = e * (-P["pitch_amp"] * rot
                                              + P["pitch_bias"] + u[3] + u[4])
        d.ctrl[aid["wing_pitch_right"]] = e * (-P["pitch_amp"] * rot
                                               + P["pitch_bias"] + u[3] - u[4])
        d.ctrl[aid["wing_roll_left"]] = e * P["roll_amp"] * np.sin(2 * ph2)
        d.ctrl[aid["wing_roll_right"]] = e * P["roll_amp"] * np.sin(2 * ph2)
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5 or d.qpos[2] > 40:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        zs.append(d.qpos[2])
        alive = k + 1
    srv = alive * dtp
    up = float(np.mean(ups)) if ups else 0.0
    zerr = float(np.mean(np.abs(np.array(zs) - 12.0))) if zs else 20.0
    return srv, up, zerr


def fitness(theta):
    sv, uv, ze = [], [], []
    for s in range(N_PERT):
        a, u, z = rollout(theta, pert_seed=s)
        sv.append(a)
        uv.append(u)
        ze.append(z)
    srv = float(np.mean(sv))
    up = float(np.mean(uv))
    ze = float(np.mean(ze))
    # 生存を主項、直立維持と高度保持を補助項に
    return srv + 0.5 * max(up, 0.0) - 0.02 * min(ze, 20.0), srv, up, ze


def work(th):
    try:
        return fitness(np.asarray(th))
    except Exception as e:
        print("err", repr(e), flush=True)
        return (0.0, 0.0, 0.0, 20.0)


def main():
    from multiprocessing import Pool
    dim = 12
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        th = np.array(ck["theta"])
        gen0, best, bth = ck["gen"], ck["best"], np.array(ck["best_theta"])
    else:
        th = np.zeros(dim)
        gen0, best, bth = 0, -1e9, th.copy()
    # hover_params は鋭く調整されており、σ=0.35 では全候補が基準1.32sから
    # 0.33s以下へ崩れた。翅運動学は細かく、静的トリムはさらに細かく探索する
    sig = np.array([0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                    0.03, 0.03, 0.03, 0.03, 0.03])
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    f0 = fitness(th)
    print(f"基準(hover_params, 帰還なし): 生存{f0[1]:.2f}s 直立{f0[2]:+.2f} "
          f"高度誤差{f0[3]:.1f} fit={f0[0]:.3f}", flush=True)
    if best < f0[0]:            # 基準を最良の初期値にする (改善の判定基準)
        best, bth = float(f0[0]), th.copy()
    with Pool(WORKERS) as pool:
        for gen in range(gen0, gen0 + n_gen):
            rng = np.random.default_rng(1000 + gen)
            eps = rng.standard_normal((N_PAIRS, dim))
            cands = []
            for e in eps:
                cands.append(th + sig * e)
                cands.append(th - sig * e)
            res = pool.map(work, cands)
            fit = np.array([r[0] for r in res])
            g = np.zeros(dim)
            for i, e in enumerate(eps):
                g += (fit[2 * i] - fit[2 * i + 1]) * e
            g /= (2 * N_PAIRS)
            th = th + LR * sig * g
            gi = int(np.argmax(fit))
            if fit[gi] > best:
                best, bth = float(fit[gi]), np.array(cands[gi])
            print(f"gen{gen}: best={best:.3f} genbest={fit[gi]:.3f} "
                  f"srvmax={max(r[1] for r in res):.2f}s "
                  f"mean={fit.mean():.3f}", flush=True)
            json.dump({"gen": gen + 1, "theta": th.tolist(), "best": best,
                       "best_theta": bth.tolist()}, open(CKPT, "w"))
    fb = fitness(bth)
    print(f"最良: 生存{fb[1]:.2f}s 直立{fb[2]:+.2f} 高度誤差{fb[3]:.1f}",
          flush=True)
    P, u = unpack(bth)
    print("翅運動学:", {k: round(float(P[k]), 4) for k in KEYS}, flush=True)
    print("静的トリム u:", np.round(u, 3), flush=True)
    np.savez("outputs/openloop_best.npz", theta=bth,
             **{k: P[k] for k in KEYS}, u=u)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
