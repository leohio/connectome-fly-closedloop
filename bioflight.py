#!/usr/bin/env python
"""統合31: 生物学的に実装可能な飛行制御を、翅運動学と帰還を同時最適化して作る。

統合30の定量結果: 現行の飛行解は≥6.5kHzの作動更新を要し、神経筋系(1羽ばたき1発,
202Hz)では実装不可能だった。統合31の対照: 変調なし開ループ(1.08s)のほうが、
それまで載せた全変調(0.25-0.44s)より良かった。

そこで制約を最初から神経筋系に合わせて設計し直す:
- 感覚: 体角速度の遅い成分のみ (ハルテア = 1羽ばたき周期の移動平均)、
        姿勢誤差 e_b (視覚・単眼)、高度・沈下速度 (オプティックフロー)
        禁止: 自分の羽ばたき反動振動の瞬時値
- 作動: 指令 u の更新は **1羽ばたきに1回だけ** (操舵MNは1周期1発)
- 同時最適化: 翅運動学7 + 静的トリム5 + 帰還K(5x7)+b(5) = 52次元
  (翅運動学は開ループ安定性の探索結果から温間スタート)
"""
import json
import os
import sys
import numpy as np
import mujoco
import openloop_hover as OH

CKPT = "outputs/bioflight.json"
T_EVAL = 3.0
N_PERT = 3
N_PAIRS = 10
LR = 0.35
WORKERS = 12
N_X, N_U = 7, 5
DIM = 12 + N_U * N_X + N_U


def rollout(theta, pert_seed=None, T=T_EVAL, n_per_stroke=1):
    E = OH.env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    P, u_trim = OH.unpack(theta[:12])
    K = theta[12:12 + N_U * N_X].reshape(N_U, N_X)
    bb = theta[12 + N_U * N_X:]
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
    NB = max(int((1.0 / P["freq"]) / dtp), 1)     # ハルテアの1周期移動平均
    buf = np.zeros((NB, 3))
    bi = 0
    ssum = np.zeros(3)
    hold = (1.0 / P["freq"]) / n_per_stroke       # 指令更新の間隔
    t_next = 0.0
    u_cmd = np.array(u_trim, float)
    u = np.array(u_trim, float)
    # 筋の単収縮ダイナミクス (τ=4.25ms, Azevedo 2020; 統合14で実測移植)。
    # 実際の筋は階段状には収縮しない。1羽ばたき1回の離散指令はこの一次遅れで
    # 平滑化され、階段が注入する高周波が除かれる — 生理的に必須の要素
    TAU_TW = 0.00425
    ups, zs, alive = [], [], 0
    for k in range(int(T / dtp)):
        t = k * dtp
        ssum += d.qvel[3:6] - buf[bi]
        buf[bi] = d.qvel[3:6].copy()
        bi = (bi + 1) % NB
        if t >= t_next:                            # 1羽ばたき1回だけ更新
            t_next += hold
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zc = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
            v = d.qvel[:3]
            ow = ssum / NB
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          e_b[0], e_b[1],
                          ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u_cmd = np.clip(u_trim + np.tanh(K @ x + bb) * 0.35, -0.55, 0.55)
        u += dtp * (u_cmd - u) / TAU_TW
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
    ze = float(np.mean(np.abs(np.array(zs) - 12.0))) if zs else 20.0
    return srv, up, ze


def fitness(theta):
    sv, uv, ze = [], [], []
    for s in range(N_PERT):
        a, u, z = rollout(theta, pert_seed=s)
        sv.append(a)
        uv.append(u)
        ze.append(z)
    srv, up, z = float(np.mean(sv)), float(np.mean(uv)), float(np.mean(ze))
    return srv + 0.5 * max(up, 0.0) - 0.03 * min(z, 20.0), srv, up, z


def work(th):
    try:
        return fitness(np.asarray(th))
    except Exception as e:
        print("err", repr(e), flush=True)
        return (0.0, 0.0, 0.0, 20.0)


def main():
    from multiprocessing import Pool
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        th = np.array(ck["theta"])
        gen0, best, bth = ck["gen"], ck["best"], np.array(ck["best_theta"])
    else:
        th = np.zeros(DIM)
        th[:12] = np.load("outputs/openloop_warm.npy")   # 開ループ探索の最良
        gen0, best, bth = 0, -1e9, th.copy()
    sig = np.concatenate([np.full(12, 0.03),
                          np.full(N_U * N_X + N_U, 0.05)])
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    f0 = fitness(th)
    print(f"開始(開ループ最良+帰還ゼロ): 生存{f0[1]:.2f}s 直立{f0[2]:+.2f} "
          f"高度誤差{f0[3]:.1f} fit={f0[0]:.3f}", flush=True)
    if best < f0[0]:
        best, bth = float(f0[0]), th.copy()
    with Pool(WORKERS) as pool:
        for gen in range(gen0, gen0 + n_gen):
            rng = np.random.default_rng(7000 + gen)
            eps = rng.standard_normal((N_PAIRS, DIM))
            cands = []
            for e in eps:
                cands.append(th + sig * e)
                cands.append(th - sig * e)
            res = pool.map(work, cands)
            fit = np.array([r[0] for r in res])
            g = np.zeros(DIM)
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
    np.save("outputs/bioflight_best.npy", bth)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
