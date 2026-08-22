#!/usr/bin/env python
"""統合36: 帰還則Kを、個体の飛行経験からの報酬変調学習で獲得する。

統合35で読出しは局所デルタ則になった。残る外部最適化は帰還則K(5x7)+b(5)=40次元の
ES である。ここでは ES (世代・集団・外部選抜) を捨て、**1個体が逐次の飛行
エピソードから学ぶ** node perturbation / REINFORCE (Fiete & Seung 2006) に置き換える:

  各試行対: 同じ突風条件で Θ+σξ と Θ-σξ を連続して飛び、報酬差で更新する
  報酬 R = 生存 + 0.5·直立度 - 0.03·高度誤差   (すべてハエ自身が感知できる量:
           墜落・視覚水平・オプティックフロー)
  更新: Θ += α·(R⁺ - R⁻)/(2σ)·ξ,  更新ノルムをクリップ

  初版 (コミット履歴参照) は単発REINFORCE+走行平均ベースラインで実装したが、
  (1) 実効ステップα/σ=1.33が過大でΘが数エピソードでtanh飽和域へ発散し、
  (2) 毎試行変わる突風がadvantageを支配してΘがランダムウォークした。
  8個体全員が開ループ(1.12s)より悪い0.29sに固着する完全失敗 (outputs/reward_K.json
  の初版)。対試行差分は「似た条件での連続2試行の比較」であり、局所性・スカラー報酬
  という生物学的制約は保っている。

生得とみなすもの: 翅運動学7+トリム5 (進化=外側の最適化が用意したCPG波形)。
学習で獲得するもの: 帰還ゲイン40次元、初期値ゼロ (=開ループの幼体から出発)。

問い: 個体の寿命内で妥当なエピソード数 (数百回・数十分の飛行経験) で
開ループ(約1.7s)から飛行(3.00s)に到達するか。
"""
import json
import os
import sys
import numpy as np
from multiprocessing import Pool
import bioflight as BF

TH = np.load("outputs/bioflight_best.npy")
INNATE = TH[:12].copy()          # 生得: 翅運動学+トリム
N_PAR = BF.N_U * BF.N_X + BF.N_U # 40
N_EP = int(__import__('os').environ.get('N_EP_OVERRIDE', '600'))  # 試行対の数
SIGMA = 0.05
ALPHA = 0.02
MAX_STEP = 0.05                  # 1更新のノルム上限
CHECK = 50                       # 学習曲線の評価間隔


SENSOR_G = np.array([0.80, 0.92, 0.91])   # 実測ゲイン (統合36d)
SENSOR_BIAS = np.array([-0.78, -0.91, -0.62])
SENSOR_NSTD = 1.9
SENSOR_NTAU = 0.007
SENSOR_DELAY = int(__import__('os').environ.get('SENSOR_DELAY', '0'))   # 追加の純遅延 [羽ばたき数]。実回路はシナプス・積分でさらに遅い疑い
SENSOR = "circuit_model"   # "true"=真の遅いω / "circuit_model"=回路復号の遅延・ノイズ模型


def rollout_sensed(theta, pert_seed, T=3.0, sensor=None, noise_seed=0):
    """BF.rollout と同一だが、ω感覚に回路復号の特性模型を挟める。

    統合36bの転移失敗の原因究明用: 学習Kは真の遅いωでは3.00s飛ぶが、
    実回路の復号ω (位相推定τ=10msの遅れ + 復号誤差 + 起動0.12sゲート) では
    0.37sに崩壊した。この模型でギャップが再現されるなら、感覚特性の差が原因。
    """
    import mujoco
    import openloop_hover as OH
    sensor = sensor or SENSOR
    E = OH.env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    P, u_trim = OH.unpack(theta[:12])
    K = theta[12:12 + BF.N_U * BF.N_X].reshape(BF.N_U, BF.N_X)
    bb = theta[12 + BF.N_U * BF.N_X:]
    rgn = np.random.default_rng(5150 + noise_seed)
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
    NB = max(int((1.0 / P["freq"]) / dtp), 1)
    buf = np.zeros((NB, 3))
    bi = 0
    ssum = np.zeros(3)
    hold = 1.0 / P["freq"]
    t_next = 0.0
    u_cmd = np.array(u_trim, float)
    u = np.array(u_trim, float)
    om_sen = np.zeros(3)
    TAU_TW = 0.00425
    TAU_DEC = 0.012                    # 実測の復号遅れ (統合36d)
    ups, zs, oms, alive = [], [], [], 0
    for k in range(int(T / dtp)):
        t = k * dtp
        ssum += d.qvel[3:6] - buf[bi]
        buf[bi] = d.qvel[3:6].copy()
        bi = (bi + 1) % NB
        if t >= t_next:
            t_next += hold
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zc = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
            v = d.qvel[:3]
            o_slow = ssum / NB
            if sensor == "circuit_model":
                # 統合36dの実測同定 (outputs/sensor_fit.json):
                # est = G·lowpass(true,12ms) + bias + AR(1)ノイズ(std1.9, τ7ms)
                tgt = SENSOR_G * np.clip(o_slow, -12, 12) + SENSOR_BIAS
                om_sen += hold * (tgt - om_sen) / TAU_DEC
                ar_a = np.exp(-hold / SENSOR_NTAU)
                ar_state = getattr(rollout_sensed, "_ar", None)
                nz = rgn.normal(0, SENSOR_NSTD * np.sqrt(1 - ar_a**2), 3)
                if ar_state is None or ar_state[0] != id(d):
                    ar = nz
                else:
                    ar = ar_a * ar_state[1] + nz
                rollout_sensed._ar = (id(d), ar)
                ow_now = om_sen + ar
                if SENSOR_DELAY > 0:
                    dbuf = getattr(rollout_sensed, "_dbuf", None)
                    if dbuf is None or dbuf[0] != id(d):
                        dbuf = (id(d), [np.zeros(3)] * SENSOR_DELAY)
                    ow_del = dbuf[1].pop(0)
                    dbuf[1].append(ow_now.copy())
                    rollout_sensed._dbuf = dbuf
                    ow_now = ow_del
                ow = ow_now if t > 0.12 else np.zeros(3)
            else:
                ow = o_slow
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          e_b[0], e_b[1],
                          ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u_cmd = np.clip(u_trim + np.tanh(K @ x + bb) * 0.35, -0.55, 0.55)
        u += dtp * (u_cmd - u) / TAU_TW
        amp = np.clip(1.0 + u[0], 0.5, 1.6)
        env0 = min(t / 0.03, 1.0)
        ph2 = 2 * np.pi * P["freq"] * t
        sn = np.sin(ph2)
        rot = np.tanh(kk * np.cos(ph2 + P["phase"])) / np.tanh(kk)
        e = env0 * amp
        d.ctrl[:] = 0
        d.ctrl[aid["wing_yaw_left"]] = e * (P["yaw_amp"] * sn + u[1] + u[2])
        d.ctrl[aid["wing_yaw_right"]] = e * (P["yaw_amp"] * sn + u[1] - u[2])
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
        # 体角速度の遅い成分 (復号が見る量)。瞬時値は羽ばたき反動振動
        # (±60rad/s級) が支配するため、ペナルティには使えない (統合36f)
        oms.append(float(np.linalg.norm(ssum / NB)))
        alive = k + 1
    srv = alive * dtp
    up = float(np.mean(ups)) if ups else 0.0
    ze = float(np.mean(np.abs(np.array(zs) - 12.0))) if zs else 20.0
    om_m = float(np.mean(oms)) if oms else 0.0
    return srv, up, ze, om_m


LAM_OM = 0.05   # 体角速度ペナルティ。統合36eの解剖: 学習Kは|ω|~25の飽和域で飛び
                # (復号線形域は~12、ES-Kは中央値11.5)、感覚喪失で墜落していた。
                # 高速回転は危険でハエ自身が感知できる量なので報酬に含める


def episode_reward(theta, pert_seed, noise_seed=0):
    """1エピソード = 指定の突風から3秒間飛ぶ。報酬はハエが感知できる量のみ"""
    srv, up, ze, om_m = rollout_sensed(theta, pert_seed, noise_seed=noise_seed)
    return srv + 0.5 * max(up, 0.0) - 0.03 * min(ze, 20.0) - LAM_OM * om_m


def evaluate(theta, n=2):
    """学習を止めて素のΘを固定外乱で測る (学習曲線用)"""
    return float(np.mean([rollout_sensed(theta, pert_seed=s,
                                         noise_seed=100 + s)[0]
                          for s in range(n)]))


def run(seed):
    try:
        return _run_inner(seed)
    except Exception as exc:
        import traceback
        return dict(seed=seed, failed=traceback.format_exc()[-500:],
                    curve=[], reached=None, final=0.0, theta=[])


def _run_inner(seed):
    import os
    rg = np.random.default_rng(31000 + seed)
    ft = os.environ.get("FINETUNE")
    if ft:
        import json as _j
        allr = _j.load(open(ft))
        sel = _j.load(open("outputs/reward_K_selected.json"))["selected_seed"]
        th = np.array([r for r in allr if r["seed"] == sel][0]["theta"])
    else:
        th = np.zeros(N_PAR)
    curve, reached = [], None
    for ep in range(N_EP):
        xi = rg.standard_normal(N_PAR)
        gust = int(rg.integers(0, 10**6))
        nz = int(rg.integers(0, 10**6))
        rp = episode_reward(np.concatenate([INNATE, th + SIGMA * xi]), gust,
                            noise_seed=nz)
        rm = episode_reward(np.concatenate([INNATE, th - SIGMA * xi]), gust,
                            noise_seed=nz)
        step = ALPHA * (rp - rm) / (2 * SIGMA) * xi
        nn = float(np.linalg.norm(step))
        if nn > MAX_STEP:
            step *= MAX_STEP / nn
        th = th + step
        if (ep + 1) % CHECK == 0:
            s_eval = evaluate(np.concatenate([INNATE, th]))
            curve.append((ep + 1, s_eval))
            print(f"[個体{seed}] {ep+1}対 eval={s_eval:.2f}s", flush=True)
            if reached is None and s_eval >= 2.995:
                reached = ep + 1
    return dict(seed=seed, curve=curve, reached=reached,
                final=evaluate(np.concatenate([INNATE, th]), n=4),
                theta=[float(x) for x in th])


if __name__ == "__main__":
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with Pool(n_runs) as p:
        res = p.map(run, range(n_runs))
    base = evaluate(np.concatenate([INNATE, np.zeros(N_PAR)]), n=4)
    print(f"開ループ (K=0, 幼体): 生存{base:.2f}s", flush=True)
    bad = [r for r in res if r.get("failed")]
    for r in bad:
        print(f"個体{r['seed']} 失敗: {r['failed'][-200:]}", flush=True)
    res = [r for r in res if not r.get("failed")]
    for r in res:
        tail = ", ".join(f"{e}:{s:.2f}" for e, s in r["curve"][::6])
        rc = f"{r['reached']}ep で3.00s到達" if r["reached"] else "未到達"
        print(f"個体{r['seed']}: 最終{r['final']:.2f}s  {rc}  [{tail}]",
              flush=True)
    fin = np.array([r["final"] for r in res])
    rch = [r["reached"] for r in res if r["reached"]]
    print(f"\n{len(res)}個体: 最終生存 {fin.mean():.2f}±{fin.std():.2f}s "
          f"(範囲 {fin.min():.2f}-{fin.max():.2f})", flush=True)
    print(f"3.00s到達: {len(rch)}/{len(res)}個体"
          + (f", 到達エピソード中央値 {int(np.median(rch))}" if rch else ""),
          flush=True)
    with open("outputs/reward_K.json", "w") as f:
        json.dump(res, f, indent=2)
    print("保存 outputs/reward_K.json", flush=True)
    print("DONE", flush=True)
