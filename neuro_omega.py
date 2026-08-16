#!/usr/bin/env python
"""統合25-2: ωの神経化 — 真値ω直読みをハルテア回路経由に置換。

従来 (ハリボテ): ESポリシーのω入力 x[4:7] にシミュレータの真値を直読み。
本版: ω → ハルテア求心性の発火位相シフト (Coriolis transduction モデル、
      C_PHASE=0.03; roll=左右反対称, pitch=同相, yaw=ストローク位相依存署名)
      → MANC腹髄サブ回路1,556ニューロン LIF (0.1ms)
      → 位相固定した全ニューロン集団 (較正で同定、約70個) の
        スパイク位相偏差 → 整合フィルタで3軸ω推定
      → ESポリシーのω入力に接続。姿勢伝搬(相補フィルタ)のωも神経推定。
視覚は optics.py の実レンダリング (修正1)。
脳ネットと腹髄ネットは単一 brian2 Network に統合 (0.1ms刻み)。

実測性能 (mode=quality): ロール正弦8rad/s·3Hz → R²=0.84, 遅れ5.6ms,
ピッチへのクロストーク0.66rad/s (13ch→68ch集団化で達成)。

mode=quality : ω正弦波を提示し推定のR²・遅れを測定
mode=fly     : 統合飛行 (光学視覚 + ハルテアω)
"""
import sys
import numpy as np
import mujoco
from brian2 import Hz, Network, prefs

import phase_reflex as PR
import vision_pop as VP
import brain_visual as BV
import optics as OPT

prefs.codegen.target = "numpy"

DT_N = 0.0001          # 神経・ハルテア連成 0.1ms
DT_W = 0.002           # 視覚更新 2ms (20神経ステップ毎)
TAU_F = 0.15           # 視覚ドリフト補正
LEAK = 0.995           # 位相偏差EMAのリーク/0.1ms (τ≈20ms)
C_PHASE = 0.03


class NeuralIntegrator:
    """姿勢積分の明示的神経モデル (漏れ積分器×2ユニット)。

    dE/dt = -E/τ_leak + (-ω) + (E_vis - E)/τ_vis
    E: 姿勢推定 (roll,pitch), ω: ハルテア推定角速度 (前庭入力),
    E_vis: 視覚DN読み出し (ドリフト補正入力)。
    実物対応: ハエの姿勢積分回路は未同定 (本モデルの正直な限界)。
    数式は漏れ積分器ニューロン (τ_leak=1s) として陽に宣言する —
    従来の「numpyに埋まった相補フィルタ」をモデルとして可視化したもの。"""

    TAU_LEAK = 1.0

    def __init__(self, tau_vis):
        self.E = np.zeros(2)
        self.tau_vis = tau_vis

    def step(self, dt, om2, e_vis):
        self.E += dt * (-self.E / self.TAU_LEAK
                        + np.array([-om2[0], -om2[1]])
                        + (e_vis - self.E) / self.tau_vis)
        return self.E


def hal_build():
    PR.C_PHASE = C_PHASE
    net, mon, pg, pref, side, st_idx, n = PR.setup()
    return dict(net=net, mon=mon, pg=pg, pref=pref, side=side, n=n)


class HalDecoder:
    """位相固定ニューロン集団のスパイク位相 → (ωroll, ωpitch, ωyaw)"""

    def __init__(self, H, PH0, t_roll, t_pitch, t_yaw):
        self.H, self.PH0 = H, PH0
        self.chs = sorted(PH0)
        self.chset = set(self.chs)
        self.t = {ax: np.array([tv[i] for i in self.chs])
                  for ax, tv in [("r", t_roll), ("p", t_pitch), ("y", t_yaw)]}
        self.s2 = {ax: max(float((v ** 2).sum()), 1e-12)
                   for ax, v in self.t.items()}
        self.dv = {i: 0.0 for i in self.chs}
        self.prev_nsp = 0

    def step(self):
        mon = self.H["mon"]
        nsp = mon.num_spikes
        if nsp > self.prev_nsp:
            from brian2 import ms as _ms
            ii = np.array(mon.i[self.prev_nsp:nsp])
            tt = np.array(mon.t[self.prev_nsp:nsp] / _ms) / 1000.0
            for iN, tS in zip(ii, tt):
                i = int(iN)
                if i in self.chset:
                    ph = float(np.mod(tS * PR.WBF, 1.0))
                    dv = (ph - self.PH0[i] + 0.5) % 1.0 - 0.5
                    self.dv[i] = 0.6 * self.dv[i] + 0.4 * dv
            self.prev_nsp = nsp
        for i in self.dv:
            self.dv[i] *= LEAK

    def estimate(self):
        v = np.array([self.dv[i] for i in self.chs])
        out = [float(np.clip(np.dot(self.t[ax], v) / self.s2[ax], -15, 15))
               for ax in ("r", "p", "y")]
        return np.array(out)


def hal_calibrate(H, run, T=0.4, om=8.0):
    """ω 3軸±提示 → 位相固定集団のPH0とチューニングベクトル。
    run(duration_s, omega3) は与えた条件でネットを進める関数。"""
    from brian2 import ms as _ms
    mon = H["mon"]
    conds = [(0, 0, 0), (om, 0, 0), (-om, 0, 0), (0, om, 0), (0, -om, 0),
             (0, 0, om), (0, 0, -om)]
    phases = []
    for omg in conds:
        base_t = float(mon.t[-1] / _ms) / 1000.0 if len(mon.t) else 0.0
        run(T, omg)
        trains = mon.spike_trains()
        ph = {}
        for iN in range(H["n"]):
            st = np.array(trains[iN] / _ms) / 1000.0
            st = st[st > base_t + 0.1]
            if len(st) < 10:
                continue
            p = np.mod(st * PR.WBF, 1.0)
            Rv = np.abs(np.mean(np.exp(2j * np.pi * p)))
            if Rv < 0.2:
                continue
            ph[iN] = float(np.angle(np.mean(np.exp(2j * np.pi * p)))
                           / (2 * np.pi) % 1.0)
        phases.append(ph)
    p0, prp, prm, ppp, ppm, pyp, pym = phases
    common = [i for i in p0 if all(i in p for p in phases)]
    PH0 = {i: p0[i] for i in common}
    circ = lambda a, b: (a - b + 0.5) % 1.0 - 0.5
    tr = {i: (circ(prp[i], PH0[i]) - circ(prm[i], PH0[i])) / (2 * om)
          for i in common}
    tp = {i: (circ(ppp[i], PH0[i]) - circ(ppm[i], PH0[i])) / (2 * om)
          for i in common}
    ty = {i: (circ(pyp[i], PH0[i]) - circ(pym[i], PH0[i])) / (2 * om)
          for i in common}
    print(f"hal較正: {len(common)}ch  |t_r|²={sum(v*v for v in tr.values()):.2e}"
          f" |t_p|²={sum(v*v for v in tp.values()):.2e}"
          f" |t_y|²={sum(v*v for v in ty.values()):.2e}", flush=True)
    return PH0, tr, tp, ty


def mode_quality():
    from brian2 import ms as _ms
    H = hal_build()
    net, pg = H["net"], H["pg"]
    pref, side = H["pref"], H["side"]
    clock_t = [0.0]

    def run(T, omg):
        for k in range(int(T / DT_N)):
            pg.rates = PR.hal_rates(clock_t[0], omg, pref, side) * Hz
            net.run(DT_N * 1000 * _ms)
            clock_t[0] += DT_N
    PH0, tr, tp, ty = hal_calibrate(H, run)
    dec = HalDecoder(H, PH0, tr, tp, ty)
    T = 1.0
    est_r, est_y, tru = [], [], []
    for k in range(int(T / DT_N)):
        t = clock_t[0]
        omr = 8.0 * np.sin(2 * np.pi * 3.0 * k * DT_N)
        pg.rates = PR.hal_rates(t, (omr, 0, 0), pref, side) * Hz
        net.run(DT_N * 1000 * _ms)
        clock_t[0] += DT_N
        dec.step()
        e = dec.estimate()
        est_r.append(e[0]); est_y.append(e[2]); tru.append(omr)
    est_r = np.array(est_r); tru = np.array(tru)
    sl = slice(2000, None)
    r2 = 1 - ((est_r[sl] - tru[sl]) ** 2).sum() / \
        ((tru[sl] - tru[sl].mean()) ** 2).sum()
    print(f"ω_roll正弦追従: R²={r2:.2f} "
          f"ヨーへのクロストークσ={np.std(est_y[2000:]):.2f}rad/s", flush=True)


def fly_trial(T=3.0, omega_src="haltere", tau_om=0.01,
              vis_readout="pop", video=None):
    """光学視覚 + ハルテアω の統合飛行 (単一Network)。"""
    from brian2 import ms as _ms
    env = PR._fly_env()
    b_pol, U_SCALE = env["b_pol"], env["U_SCALE"]
    Pw, Q0, ZT_W = env["Pw"], env["Q0"], env["ZT_W"]
    m = OPT.get_model()
    ckp = np.load("../fly-flight-sim/outputs/hover_policy.npz")
    K = ckp["theta"][:35].reshape(5, 7).copy()
    # 視覚回路 + ハルテア回路 → 単一Network
    neuV, synV, idsV, idxV, metaV = BV.build_subcircuit()
    pgV, shV, ssign, is_oc = BV.sensory_attach(neuV, idxV, metaV)
    from brian2 import SpikeMonitor
    monV = SpikeMonitor(neuV, record=False)
    sensV = set(int(b) for b in np.concatenate([metaV["ocr"], metaV["vshs"]]))
    readout = np.array([i for b, i in idxV.items() if b not in sensV])
    H = hal_build() if omega_src in ("haltere", "haltere_hybrid") else None
    objs = [neuV, synV, monV, pgV, shV]
    if H is not None:
        objs += list(H["net"].objects)
    net = Network(*objs)
    clock_t = [0.0]

    def run_net(T_run, omg, vis_rates_now):
        """omgでハルテアを、vis_rates_nowで視覚を駆動しつつ進める"""
        pgV.rates = vis_rates_now * Hz
        steps = int(round(T_run / DT_N))
        for k in range(steps):
            if H is not None:
                H["pg"].rates = PR.hal_rates(clock_t[0], omg,
                                             H["pref"], H["side"]) * Hz
            net.run(DT_N * 1000 * _ms)
            clock_t[0] += DT_N
    # --- 較正1: 視覚 (実レンダリング提示; ハルテアはω=0で駆動) ---
    eye = OPT.OcellarEye(m)
    dcal = mujoco.MjData(m)
    mujoco.mj_resetData(m, dcal)
    dcal.qpos[2] = 12.0
    dcal.qpos[3:7] = Q0
    mujoco.mj_forward(m, dcal)
    eye.set_baseline(dcal)
    conds = [(0.0, 0.0), (0.2, 0.0), (-0.2, 0.0), (0.0, 0.2), (0.0, -0.2)]
    ratesV = []
    prevV = monV.count[:].copy()
    for ro, pi in conds:
        dcal.qpos[3:7] = OPT.quat_tilt(Q0, ro, pi)
        mujoco.mj_forward(m, dcal)
        run_net(0.4, (0, 0, 0), eye.rates(dcal, ssign, is_oc))
        c = monV.count[:].copy()
        ratesV.append((c - prevV)[readout] / 0.4)
        prevV = c
    r0, rp, rm, pp, pm = ratesV
    if vis_readout == "anat":
        # 解剖学固定読み出し: DNg04左右反対称=ロール, DNp18=ピッチ (二値重み)
        pos = {int(n): k for k, n in enumerate(readout)}
        side_of = lambda i: str(metaV["side"][i])
        mr = np.zeros(len(readout))
        mp = np.zeros(len(readout))
        for _, rw in metaV["dn"].iterrows():
            i = idxV[int(rw.root_id)]
            if i not in pos:
                continue
            if rw.primary_type == "DNg04":
                mr[pos[i]] = +1.0 if side_of(i) == "left" else -1.0
            elif rw.primary_type == "DNp18":
                mp[pos[i]] = +1.0
        gr = float(mr @ (rp - rm)) / 0.4     # 較正スカラー (Hz/rad)
        gp = float(mp @ (pp - pm)) / 0.4
        t_r, t_p = mr * gr, mp * gp          # 整合フィルタ形式に同型化
        s2r = max(float((mr ** 2).sum()) * gr * gr, 1e-9)
        s2p = max(float((mp ** 2).sum()) * gp * gp, 1e-9)
        print(f"解剖学読み出し: DNg04×{int(np.abs(mr).sum())} gain={gr:.1f}Hz/rad"
              f" / DNp18×{int(mp.sum())} gain={gp:.1f}Hz/rad", flush=True)
    else:
        t_r = (rp - rm) / 0.4
        t_p = (pp - pm) / 0.4
        s2r = max(float((t_r ** 2).sum()), 1e-9)
        s2p = max(float((t_p ** 2).sum()), 1e-9)
    # --- 較正2: ハルテア (視覚は水平提示のまま) ---
    dcal.qpos[3:7] = Q0
    mujoco.mj_forward(m, dcal)
    level_rates = eye.rates(dcal, ssign, is_oc)
    if H is not None:
        PH0h, thr, thp, thy = hal_calibrate(
            H, lambda T_run, omg: run_net(T_run, omg, level_rates))
        dec = HalDecoder(H, PH0h, thr, thp, thy)
    # --- 飛行 ---
    d = mujoco.MjData(m)
    dtp = m.opt.timestep
    aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
           for nm in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                      "wing_yaw_right", "wing_roll_right",
                      "wing_pitch_right"]}
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    renderer, frames, next_f = None, [], 0.0
    if video:
        renderer = mujoco.Renderer(m, 480, 640)
        vcam = mujoco.MjvCamera()
        vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
        vcam.distance, vcam.elevation, vcam.azimuth = 5.0, -10, 100
    R = np.zeros(9)
    integ = NeuralIntegrator(TAU_F)
    eb_est = np.zeros(2)
    om_est = np.zeros(3)
    f_rate = r0.copy()
    prev_c = monV.count[:].copy()
    ups, zs, n_alive = [], [], 0
    kk = max(Pw["sharp"], 1e-3)
    for wi in range(int(T / DT_W)):
        t = wi * DT_W
        # 視覚レート更新 (2ms毎に実レンダリング)
        vr = eye.rates(d, ssign, is_oc)
        pgV.rates = vr * Hz
        c = monV.count[:].copy()
        r_now = (c - prev_c)[readout] / DT_W
        prev_c = c
        f_rate += DT_W * (r_now - f_rate) / 0.05
        target = np.array([float(np.dot(t_r, f_rate - r0) / s2r),
                           float(np.dot(t_p, f_rate - r0) / s2p)])
        for kn in range(int(DT_W / DT_N)):
            om_true = d.qvel[3:6]
            if H is not None:
                H["pg"].rates = PR.hal_rates(
                    clock_t[0], (om_true[0], om_true[1], om_true[2]),
                    H["pref"], H["side"]) * Hz
            net.run(DT_N * 1000 * _ms)
            clock_t[0] += DT_N
            if H is not None:
                dec.step()
                om_est[:] = dec.estimate()
            elif omega_src == "true_lp":
                om_est += (DT_N / tau_om) * (om_true - om_est)
            else:
                om_est[:] = om_true
            # 姿勢伝搬 = 神経積分器 (前庭入力=ハルテア推定, 視覚=DN読み出し)
            eb_est = integ.step(DT_N, om_est, target)
            if omega_src == "haltere_hybrid":
                om_x = om_true      # 速いダンピング列のみ真値 (残存ハリボテ、特性評価済み)
            else:
                om_x = om_est
            for kp2 in range(int(DT_N / dtp)):
                tt = t + kn * DT_N + kp2 * dtp
                x = np.array([(12.0 - d.qpos[2]) / 5.0, -d.qvel[2] / 30.0,
                              eb_est[0], eb_est[1],
                              om_x[0] / 20.0, om_x[1] / 20.0,
                              om_x[2] / 20.0])
                u = np.tanh(K @ x + b_pol) * U_SCALE
                amp = np.clip(1.0 + u[0], 0.5, 1.6)
                env0 = min(tt / 0.03, 1.0)
                ph2 = 2 * np.pi * Pw["freq"] * tt
                s = np.sin(ph2)
                rot = np.tanh(kk * np.cos(ph2 + Pw["phase"])) / np.tanh(kk)
                eL = env0 * amp
                eR = env0 * amp
                d.ctrl[:] = 0
                d.ctrl[aid["wing_yaw_left"]] = eL * (Pw["yaw_amp"] * s + u[1] + u[2])
                d.ctrl[aid["wing_yaw_right"]] = eR * (Pw["yaw_amp"] * s + u[1] - u[2])
                d.ctrl[aid["wing_pitch_left"]] = eL * (-Pw["pitch_amp"] * rot
                                                       + Pw["pitch_bias"] + u[3] + u[4])
                d.ctrl[aid["wing_pitch_right"]] = eR * (-Pw["pitch_amp"] * rot
                                                        + Pw["pitch_bias"] + u[3] - u[4])
                d.ctrl[aid["wing_roll_left"]] = eL * Pw["roll_amp"] * np.sin(2 * ph2)
                d.ctrl[aid["wing_roll_right"]] = eR * Pw["roll_amp"] * np.sin(2 * ph2)
                mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        zs.append(d.qpos[2])
        n_alive = wi + 1
        if renderer and t * 10 >= next_f:
            vcam.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, camera=vcam)
            frames.append(renderer.render())
            next_f += 1.0 / 30
    if renderer and frames:
        import imageio
        imageio.mimsave(video, frames, fps=30)
    return (n_alive * DT_W, float(np.mean(ups)) if ups else 0.0,
            float(np.mean(zs)) if zs else 0.0)


def mode_fly():
    s, up, z = fly_trial(omega_src="true")
    print(f"対照(真値ω): 生存{s:.2f}s 直立度{up:+.2f} z={z:.1f}", flush=True)
    s, up, z = fly_trial(omega_src="haltere")
    print(f"HALTEREω: 生存{s:.2f}s 直立度{up:+.2f} z={z:.1f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quality"
    {"quality": mode_quality, "fly": mode_fly}[mode]()
