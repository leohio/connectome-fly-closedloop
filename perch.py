#!/usr/bin/env python
"""統合26 LoopD-F: 木を見つけ、近づき、触れて、止まる。

チェーン:
  複眼レンダリング → 光受容体11,151 (実座標レチノトピー) → 実配線 →
  LC4/LPLC2 (抑制符号の物体信号) → 方位=左右差 → ヨーレート指令 (視覚servo)
                                → 接近度=総抑制 → 減速判断
  接触 (MuJoCo実接触) → 前脚触覚求心性533 → MANC全腹髄23,188 →
  ネットワーク応答140倍 → 翅停止判断
飛行基盤: 速度指令型ESポリシー (統合26 LoopAで再訓練、工学層)。

判断則2つ (θ_app: 減速, θ_touch: 翅停止) は工学層として明示。
感知と伝搬はすべて実コネクトーム回路。
"""
import sys
import numpy as np
import mujoco
from brian2 import Hz, ms, Network, defaultclock, prefs

import phase_reflex as PR
import object_vision as OV
import vnc_model

prefs.codegen.target = "numpy"

VIS_DT = 0.010        # 判断ループ 10ms
V_APPROACH = 2.0      # 接近速度指令 (権限弱・ドリフト主体)
K_PSI = 1.5           # 方位信号→進行方位指令の積分ゲイン
TH_APP = 0.15         # 減速判断の抑制閾値
TH_TOUCH = 5.0        # 触覚判断: VNC応答倍率
TREE = (-18.0, 8.0)
Z_TGT = 20.0   # 高度目標 (実現高度はサグにより約8-12)


def run_demo(T=8.0, video=None, verbose=True, psi0=0.0,
             kpsi=1.5):
    defaultclock.dt = 1 * ms      # 判断回路は1ms粒度で十分
    env = PR._fly_env()
    b_pol = env["b_pol"]
    Pw, Q0 = env["Pw"], env["Q0"]
    _R0 = np.zeros(9)
    mujoco.mju_quat2Mat(_R0, np.array(Q0))
    ZT_W = np.array([_R0[2], _R0[5], _R0[8]])
    m = OV.get_tree_model(tree=TREE, radius=3.0, height=30.0)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = (2.0, 0.0, 12.0)
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    # ナビポリシー (5x9)
    ck = np.load("../fly-flight-sim/outputs/hover_policy.npz")
    K = ck["theta"][:35].reshape(5, 7)     # 元の強い安定化ポリシー
    bp = ck["theta"][35:]
    U_SCALE = np.array([0.5, 0.5, 0.4, 0.4, 0.3])
    aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
           for nm in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                      "wing_yaw_right", "wing_roll_right",
                      "wing_pitch_right"]}
    tree_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tree")
    adh_ids = [i for i in range(m.nu)
               if "adhere" in (mujoco.mj_id2name(
                   m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")]
    for ai in adh_ids:
        m.actuator_gainprm[ai, 0] *= 20.0    # 衝突着地の運動量に耐える把持力
    for g in range(m.ngeom):
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "claw" in gn and "collision" in gn:
            m.geom_margin[g] = 0.03          # 爪の吸着発動域
    # --- 神経回路: 物体視 + 腹髄触覚 (単一Network) ---
    eye = OV.CompoundEye(m)
    netO, monO, pgO, pix, r_side, groups, nO = OV.build_object_net()
    props_sn = __import__("pandas").read_feather("neuron-properties.feather")
    sn = props_sn[(props_sn["class"] == "sensory neuron")
                  & (props_sn.modality == "tactile")
                  & props_sn.entryNerve.isin(["ProLN_L", "ProLN_R"])]
    tact = [int(b) for b in sn.bodyId]
    netV, monV, idsV, idxV, pgV = vnc_model.make_network(
        [], r_stim_hz=0, sensory_bodyids=tact)
    keptV = [b for b in tact if b in idxV]
    net = Network(*netO.objects, *netV.objects)
    nV = len(idsV)
    # --- 視覚基線較正 (開始位置で木に背を向けた静止ポーズ) ---
    bearing0 = np.arctan2(TREE[1] - d.qpos[1], TREE[0] - d.qpos[0])
    dcal = mujoco.MjData(m)
    mujoco.mj_resetData(m, dcal)
    dcal.qpos[:3] = d.qpos[:3].copy()
    yaw = bearing0                      # 基線 = 開始位置で木の方向 (遠方=微小)
    dq = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, dq, np.array(Q0))
    dcal.qpos[3:7] = out
    mujoco.mj_forward(m, dcal)
    pgO.rates = OV.phot_rates(eye.images(dcal), pix, r_side) * Hz
    pgV.rates = np.zeros(len(keptV)) * Hz
    net.run(300 * ms)
    c0 = monO.count[:].copy()
    cV0 = monV.count[:].copy()
    prevO = c0.copy()
    prevV = cV0.copy()
    # 対象群の基線
    gsel = {k: groups[k] for k in
            ["LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R"]}
    base = {k: max(c0[v].mean() / 0.3, 1.0) for k, v in gsel.items()}
    vnc_base = max((cV0.sum()) / 0.3, 1.0)
    if verbose:
        print(f"視覚基線: " + " ".join(f"{k}={base[k]:.0f}" for k in base)
              + f" / VNC基線={vnc_base:.0f}sp/s", flush=True)
    # 開始姿勢: ポリシー固有の基準方位のまま (建て直しトランジェント回避)。
    # 操舵は目標姿勢ベクトルの回転 zt=Rz(ψ)·ZT_W で行う (発見:
    # ポリシーは世界固定方位を保持するため、ヨー注入では操舵できない)
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    psi_cmd = psi0     # 事前方位補正 (グリッド較正)
    kpsi_eff = K_PSI if kpsi is None else kpsi
    # --- レンダラ ---
    renderer, frames, next_f = None, [], 0.0
    if video:
        renderer = mujoco.Renderer(m, 480, 640)
        vcam = mujoco.MjvCamera()
        vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
        vcam.distance, vcam.elevation, vcam.azimuth = 10.0, -12, 156 + 90
    # --- 状態機械 ---
    dtp = m.opt.timestep
    kk = max(Pw["sharp"], 1e-3)
    R = np.zeros(9)
    state = "APPROACH"
    cpsi, spsi = 1.0, 0.0
    touch_latch = False
    wings_on = True
    t_touch = None
    az_sig = app_sig = 0.0
    vcmd = np.zeros(2)
    Rm = np.zeros(9)
    n_steps = int(T / dtp)
    vis_every = int(VIS_DT / dtp)
    log_lines = []
    dmin = 1e9
    for k in range(n_steps):
        t = k * dtp
        # ---- 判断ループ (10ms毎) ----
        if k % vis_every == 0:
            im = eye.images(d)
            pgO.rates = OV.phot_rates(im, pix, r_side) * Hz
            pgV.rates = (np.full(len(keptV), 150.0) if touch_latch
                         else np.zeros(len(keptV))) * Hz
            touch_latch = False
            net.run(VIS_DT * 1000 * ms)   # 10ms (統合18型の単位バグを修正)
            cO = monO.count[:].copy()
            cV = monV.count[:].copy()
            drO = (cO - prevO) / VIS_DT
            vnc_rate = (cV - prevV).sum() / VIS_DT
            prevO, prevV = cO, cV
            if t < 0.3:                 # 飛行中の実測基線を収集
                for kx, vv in gsel.items():
                    base[kx] = 0.7 * base[kx] + 0.3 * max(drO[vv].mean(), 1.0)
            sup = {kx: 1.0 - drO[v].mean() / base[kx]
                   for kx, v in gsel.items()}
            az_sig = 0.5 * az_sig + 0.5 * ((sup["LC4_L"] + sup["LPLC2_L"])
                                           - (sup["LC4_R"] + sup["LPLC2_R"]))
            app_sig = 0.8 * app_sig + 0.2 * 0.5 * sum(sup.values())
            touch_sig = vnc_rate / vnc_base
            # 状態遷移
            if state == "APPROACH" and t > 0.8 and app_sig > TH_APP:
                state = "BRAKE"
                log_lines.append(f"t={t:.2f} 減速判断 (app={app_sig:.2f})")
            if state in ("APPROACH", "BRAKE") and touch_sig > TH_TOUCH:
                state = "LANDED"
                wings_on = False
                t_touch = t
                log_lines.append(f"t={t:.2f} 接触→翅停止+爪吸着ON "
                                 f"(VNC応答×{touch_sig:.0f})")
            # 誘導
            mujoco.mju_quat2Mat(Rm, d.qpos[3:7])
            head_w = -np.array([Rm[0], Rm[3], Rm[6]])   # 頭方向 ≈ 体-x
            head_w[2] = 0
            nh = np.linalg.norm(head_w) + 1e-9
            head_w /= nh
            if t >= 0.15 and state == "APPROACH":
                psi_cmd += np.clip(kpsi_eff * az_sig, -1.0, 1.0) * VIS_DT
                psi_cmd = float(np.clip(psi_cmd, -1.0, 1.0))
        # ---- 飛行制御 (毎ステップ) ----
        cpsi, spsi = np.cos(psi_cmd), np.sin(psi_cmd)
        zt = np.array([cpsi * ZT_W[0] - spsi * ZT_W[1],
                       spsi * ZT_W[0] + cpsi * ZT_W[1], ZT_W[2]])
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        zc = np.array([R[2], R[5], R[8]])
        e_b = R.reshape(3, 3).T @ np.cross(zc, zt)
        om = d.qvel[3:6]
        vel = d.qvel[:3]
        x = np.array([(12.0 - d.qpos[2]) / 5.0, -vel[2] / 30.0,
                      e_b[0], e_b[1], om[0] / 20.0, om[1] / 20.0,
                      om[2] / 20.0])
        u = np.tanh(K @ x + bp) * U_SCALE
        amp = np.clip(1.0 + u[0], 0.5, 1.6)
        env0 = min(t / 0.03, 1.0) if wings_on else 0.0
        ph2 = 2 * np.pi * Pw["freq"] * t
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
        if state in ("BRAKE", "LANDED"):
            for ai in adh_ids:
                d.ctrl[ai] = 1.0    # looming→着地準備(吸着武装)/接触→把持継続
        mujoco.mj_step(m, d)
        # 接触は毎ステップ検出してラッチ (瞬間接触の取りこぼし防止)
        if not touch_latch:
            for ci in range(d.ncon):
                g1i, g2i = d.contact[ci].geom1, d.contact[ci].geom2
                if g1i != tree_gid and g2i != tree_gid:
                    continue
                other = g2i if g1i == tree_gid else g1i
                onm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                        int(other)) or ""
                if "wing" in onm:
                    continue    # 翅接触は脚触覚(ProLN)ではない → 判断に使わない
                if True:
                    touch_latch = True
                    if verbose:
                        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                               int(d.contact[ci].geom1))
                        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                               int(d.contact[ci].geom2))
                        dd = np.hypot(d.qpos[0]-TREE[0], d.qpos[1]-TREE[1])
                        print(f"  [latch] t={t:.3f} {g1}<->{g2} "
                              f"dist={dd:.2f} z={d.qpos[2]:.2f} "
                              f"cdist={d.contact[ci].dist:.4f}", flush=True)
                    break
        zlim = 0.05 if state == "LANDED" else 0.3
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < zlim:
            break
        dnow = np.hypot(d.qpos[0]-TREE[0], d.qpos[1]-TREE[1])
        dmin = min(dmin, dnow)
        if dnow > 45:
            break                        # すり抜け失敗の早期終了
        if renderer and t * 10 >= next_f:
            vcam.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, camera=vcam)
            frames.append(renderer.render())
            next_f += 1.0 / 30
        if t_touch is not None and t - t_touch > 1.5:
            break
        if verbose and k % 20000 == 0:
            dist = np.hypot(d.qpos[0]-TREE[0], d.qpos[1]-TREE[1])
            print(f"  t={t:.2f} {state} 距離{dist:.1f} az={az_sig:+.2f} "
                  f"app={app_sig:.2f} z={d.qpos[2]:.1f}", flush=True)
    if renderer and frames:
        import imageio
        imageio.mimsave(video, frames, fps=30)
    dist = np.hypot(d.qpos[0]-TREE[0], d.qpos[1]-TREE[1])
    touching_end = any((d.contact[i].geom1 == tree_gid or
                        d.contact[i].geom2 == tree_gid)
                       for i in range(d.ncon))
    spd = float(np.linalg.norm(d.qvel[:3]))
    if state == "LANDED" and touching_end and spd < 2.0 and d.qpos[2] > 0.5:
        state = "PERCHED"
        log_lines.append(f"幹に静止: z={d.qpos[2]:.2f} 速度{spd:.2f} 接触維持")
    for ln in log_lines:
        print(ln, flush=True)
    print(f"終了: state={state} 最終距離{dist:.1f} 最接近{dmin:.1f} "
          f"t={k*dtp:.2f}s z={d.qpos[2]:.1f}", flush=True)
    return state, dmin


if __name__ == "__main__":
    vid = "outputs/perch_demo.mp4" if "video" in sys.argv else None
    run_demo(video=vid)
