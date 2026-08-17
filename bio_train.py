#!/usr/bin/env python
"""統合30: 生物学的に実現可能な感覚だけで飛行制御を訓練する。

統合29で、教師の3.0sは自身の羽ばたき反動振動 (±600rad/s, 202Hz) を
無限帯域ジャイロで読むことに依存し、神経系には実装不可能と判明した。
そこで感覚を生物学的に利用可能なものだけに制限して訓練し直し、
「実現可能な上限」を求める。これが実回路を比べるべき正しい基準になる。

利用可能とする感覚 (すべて実バエが持つ):
- 体の角速度の遅い成分 (ハルテア; 1羽ばたき周期の移動平均 = 周波数選択的復調)
- 姿勢誤差 e_b (視覚・単眼)
- 高度・沈下速度 (視覚のオプティックフロー)
禁止: 自分の羽ばたき反動振動の瞬時値
"""
import json
import os
import sys
import numpy as np
import mujoco

# 状態: 従来7 + ストローク位相の高調波6 = 13。
# 位相クロックは生物学的に正当: ハルテアは翅と同周波数・逆位相で拍動し、
# 翅ヒンジの固有受容器 (cs, chordotonal) もストローク位相を符号化する。
# 統合29の発見 — 教師のω項は安定化帰還ではなく「周期内の翅波形」の生成器で、
# 体の羽ばたき振動を位相クロックとして使っていた。ならば位相を直接与えるのが
# 生物学的に正しい定式化になる (振動の瞬時値を読む必要はない)
N_X, N_U = 13, 5
CKPT = "outputs/bio_policy.json"
T_EVAL = 3.0
N_PERT = 3
N_PAIRS = 8
SIG = 0.10
LR = 0.4
WORKERS = 12
_E = {}


def env():
    if _E:
        return _E
    import phase_reflex as PR
    import connectome_fastloop as CF
    e = PR._fly_env()
    m = CF.OPT.get_model()
    _E.update(Pw=e["Pw"], Q0=e["Q0"], ZT_W=e["ZT_W"], US=e["U_SCALE"], m=m)
    _E["aid"] = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                 for n in ["wing_yaw_left", "wing_roll_left",
                           "wing_pitch_left", "wing_yaw_right",
                           "wing_roll_right", "wing_pitch_right"]}
    return _E


def rollout(theta, pert_seed=None, T=T_EVAL):
    E = env()
    Pw, Q0, ZT_W, US, m, aid = (E["Pw"], E["Q0"], E["ZT_W"], E["US"],
                                E["m"], E["aid"])
    K = theta[:N_U * N_X].reshape(N_U, N_X)
    b = theta[N_U * N_X:N_U * N_X + N_U]
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    if pert_seed is not None:
        d.qvel[3:6] = np.random.default_rng(pert_seed).normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
    dtp = m.opt.timestep
    kk = max(Pw["sharp"], 1e-3)
    R = np.zeros(9)
    NB = max(int((1.0 / Pw["freq"]) / dtp), 1)   # 1羽ばたき周期の移動平均
    buf = np.zeros((NB, 3))
    bi = 0
    ssum = np.zeros(3)
    ups, alive = [], 0
    for k in range(int(T / dtp)):
        t = k * dtp
        ssum += d.qvel[3:6] - buf[bi]
        buf[bi] = d.qvel[3:6].copy()
        bi = (bi + 1) % NB
        ow = ssum / NB                      # ハルテアが出せる遅い角速度のみ
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        zc = np.array([R[2], R[5], R[8]])
        e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
        v = d.qvel[:3]
        phc = 2 * np.pi * Pw["freq"] * t
        x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                      e_b[0], e_b[1], ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0,
                      np.sin(phc), np.cos(phc),
                      np.sin(2 * phc), np.cos(2 * phc),
                      np.sin(3 * phc), np.cos(3 * phc)])
        u = np.tanh(K @ x + b) * US
        amp = np.clip(1.0 + u[0], 0.5, 1.6)
        env0 = min(t / 0.03, 1.0)
        ph2 = 2 * np.pi * Pw["freq"] * t
        s = np.sin(ph2)
        rot = np.tanh(kk * np.cos(ph2 + Pw["phase"])) / np.tanh(kk)
        e = env0 * amp
        d.ctrl[:] = 0
        d.ctrl[aid["wing_yaw_left"]] = e * (Pw["yaw_amp"] * s + u[1] + u[2])
        d.ctrl[aid["wing_yaw_right"]] = e * (Pw["yaw_amp"] * s + u[1] - u[2])
        d.ctrl[aid["wing_pitch_left"]] = e * (-Pw["pitch_amp"] * rot
                                              + Pw["pitch_bias"] + u[3] + u[4])
        d.ctrl[aid["wing_pitch_right"]] = e * (-Pw["pitch_amp"] * rot
                                               + Pw["pitch_bias"] + u[3] - u[4])
        d.ctrl[aid["wing_roll_left"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
        d.ctrl[aid["wing_roll_right"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        alive = k + 1
    return alive * dtp, float(np.mean(ups)) if ups else 0.0


def fitness(theta):
    sv, uv = [], []
    for s in range(N_PERT):
        a, u = rollout(theta, pert_seed=s)
        sv.append(a)
        uv.append(u)
    return (float(np.mean(sv)) + 0.5 * max(float(np.mean(uv)), 0.0),
            float(np.mean(sv)), float(np.mean(uv)))


def work(th):
    try:
        return fitness(np.asarray(th))
    except Exception as e:
        print("err", repr(e), flush=True)
        return (0.0, 0.0, 0.0)


def main():
    from multiprocessing import Pool
    dim = N_U * N_X + N_U
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        th = np.array(ck["theta"])
        gen0, best, bth = ck["gen"], ck["best"], np.array(ck["best_theta"])
    else:
        ckp = np.load("../fly-flight-sim/outputs/hover_policy.npz")
        old = ckp["theta"]
        Ko = old[:5 * 7].reshape(5, 7)
        K = np.zeros((N_U, N_X))
        K[:, :7] = Ko                      # 姿勢・ω部分は教師から温間スタート
        K[:, 4:7] = 0.0                    # 振動依存のω列は捨てる
        th = np.concatenate([K.reshape(-1), old[5 * 7:5 * 7 + 5]])
        gen0, best, bth = 0, -1.0, th.copy()
    rng = np.random.default_rng(1)
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    f0 = fitness(th)
    print(f"開始: 生存{f0[1]:.2f}s 直立{f0[2]:+.2f} fit={f0[0]:.3f}", flush=True)
    with Pool(WORKERS) as pool:
        for gen in range(gen0, gen0 + n_gen):
            eps = rng.standard_normal((N_PAIRS, dim))
            cands = []
            for e in eps:
                cands.append(th + SIG * e)
                cands.append(th - SIG * e)
            res = pool.map(work, cands)
            fit = np.array([r[0] for r in res])
            g = np.zeros(dim)
            for i, e in enumerate(eps):
                g += (fit[2 * i] - fit[2 * i + 1]) * e
            g /= (2 * N_PAIRS)
            th = th + LR * SIG * g
            gi = int(np.argmax(fit))
            if fit[gi] > best:
                best, bth = float(fit[gi]), np.array(cands[gi])
            print(f"gen{gen}: best={best:.3f} genbest={fit[gi]:.3f} "
                  f"mean={fit.mean():.3f} "
                  f"srvmax={max(r[1] for r in res):.2f}s", flush=True)
            json.dump({"gen": gen + 1, "theta": th.tolist(), "best": best,
                       "best_theta": bth.tolist()}, open(CKPT, "w"))
    fc, fb = fitness(th), fitness(bth)
    print(f"center: 生存{fc[1]:.2f}s 直立{fc[2]:+.2f}", flush=True)
    print(f"best  : 生存{fb[1]:.2f}s 直立{fb[2]:+.2f}", flush=True)
    np.save("outputs/bio_policy_best.npy", bth)


if __name__ == "__main__":
    main()
