#!/usr/bin/env python
"""統合25-1: 視覚の光学化 — 真値注入を実レンダリングに置換。

従来 (ハリボテ): vis_rates(e_b) が真の姿勢を式でスパイクに符号化 = 循環。
本版: MuJoCoシーンに非衝突の明るい床平面 (contype=0, 物理に無影響) を追加し、
3つのオセリカメラ (左上/右上/前方) で32x32を実レンダリング →
平均輝度 → 網膜杆細胞レートに変換。姿勢情報は「水平線の見え方」から
光学的にのみ得られる。実測: ロール±0.2radで左右輝度差±103 (反対称)、
ピッチは中央+左右和が単調変化 (中央はロール不変)。

残る透過モデル (正直な限界): 輝度→レートの変換式 r=R0(1+KC(b/b0−1)) と
そのゲイン KC は光受容体の transduction モデル (どのシミュレーションでも
物理刺激→受容器電流の段はモデルにならざるを得ない)。
"""
import os
import numpy as np
import mujoco

R0 = 150.0     # 網膜杆細胞の基準レート [Hz]
KC = 3.0       # 光→レートの transduction ゲイン (コントラスト増幅)
RES = 32
DIRS = dict(L=[0.4, 0.8, 0.1], R=[0.4, -0.8, 0.1], M=[1.0, 0.0, 0.1])

_MODEL = None


def get_model():
    """視覚用シーン (非衝突床) 付きの flybody モデル。物理パラメタは
    fly_flight2.build_model と同一 (床は contype=0 で力学に影響しない)。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    cwd = os.getcwd()
    os.chdir("../fly-flight-sim")
    try:
        xml = open("flybody/flybody/fruitfly/assets/fruitfly.xml").read()
        xml = xml.replace(
            'name="wing_left_fluid"',
            'name="wing_left_fluid" fluidshape="ellipsoid" '
            'fluidcoef="1.0 0.5 1.5 1.7 1.0"')
        xml = xml.replace(
            'name="wing_right_fluid"',
            'name="wing_right_fluid" fluidshape="ellipsoid" '
            'fluidcoef="1.0 0.5 1.5 1.7 1.0"')
        xml = xml.replace(
            '<worldbody>',
            '<worldbody><geom name="vision_floor" type="plane" '
            'size="80 80 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1" '
            'contype="0" conaffinity="0"/>', 1)
        assets = {os.path.basename(p): open(os.path.join(
            "flybody/flybody/fruitfly/assets", p), "rb").read()
            for p in os.listdir("flybody/flybody/fruitfly/assets")
            if p.endswith((".obj", ".png", ".stl", ".msh"))}
        m = mujoco.MjModel.from_xml_string(xml, assets)
    finally:
        os.chdir(cwd)
    m.opt.timestep = 5e-5
    for n in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
              "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]:
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        m.actuator_gainprm[aid, 0] = 18.0
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        m.jnt_stiffness[jid] = 0.01
        m.dof_damping[m.jnt_dofadr[jid]] = 0.00776923
    _MODEL = m
    return m


class OcellarEye:
    """3オセリの実レンダリング → 輝度 → 網膜レート"""

    def __init__(self, m):
        self.m = m
        self.ren = mujoco.Renderer(m, RES, RES)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.b0 = None

    def brightness(self, d):
        out = {}
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, d.qpos[3:7])
        Rb = Rm.reshape(3, 3)
        for k, v in DIRS.items():
            dw = Rb @ np.array(v)
            dw /= np.linalg.norm(dw)
            self.cam.lookat[:] = d.qpos[:3] + dw
            self.cam.distance = 1.0
            self.cam.azimuth = np.degrees(np.arctan2(dw[1], dw[0])) + 180.0
            self.cam.elevation = -np.degrees(np.arcsin(np.clip(dw[2], -1, 1)))
            self.ren.update_scene(d, camera=self.cam)
            out[k] = float(self.ren.render().mean())
        return out

    def set_baseline(self, d):
        self.b0 = self.brightness(d)

    def rates(self, d, ssign, is_oc):
        """網膜杆細胞レート。左細胞←左オセリ、右←右、中央←前方。
        VS/HS細胞は一定レート (光学入力なしの背景)。"""
        b = self.brightness(d)
        def enc(k):
            return np.clip(R0 * (1.0 + KC * (b[k] / max(self.b0[k], 1.0)
                                             - 1.0)), 0, 400)
        rL, rR, rM = enc("L"), enc("R"), enc("M")
        r_oc = np.where(ssign > 0, rL, np.where(ssign < 0, rR, rM))
        return is_oc * r_oc + (1.0 - is_oc) * 40.0


def quat_tilt(Q0, roll, pitch):
    q = np.array(Q0)
    for ang, ax in [(roll, [1, 0, 0]), (pitch, [0, 1, 0])]:
        if ang:
            dq = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * np.array(ax))])
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, q, dq)
            q = out.copy()
    return q


def calibrate_eye(net, mon, pg, eye, readout, m, Q0, cal_tilt=0.2, cal_t=0.4,
                  ssign=None, is_oc=None):
    """実レンダリングによる較正: 傾けた姿勢を実際にシーンに置いて提示。"""
    from brian2 import ms as _ms, Hz
    dcal = mujoco.MjData(m)
    mujoco.mj_resetData(m, dcal)
    dcal.qpos[2] = 12.0
    dcal.qpos[3:7] = Q0
    mujoco.mj_forward(m, dcal)
    eye.set_baseline(dcal)
    conds = [(0.0, 0.0), (cal_tilt, 0.0), (-cal_tilt, 0.0),
             (0.0, cal_tilt), (0.0, -cal_tilt)]
    rates = []
    prev = mon.count[:].copy()
    for ro, pi in conds:
        dcal.qpos[3:7] = quat_tilt(Q0, ro, pi)
        mujoco.mj_forward(m, dcal)
        pg.rates = eye.rates(dcal, ssign, is_oc) * Hz
        net.run(cal_t * 1000 * _ms)
        c = mon.count[:].copy()
        rates.append((c - prev)[readout] / cal_t)
        prev = c
    r0, rp, rm, pp, pm = rates
    t_roll = (rp - rm) / (2 * cal_tilt)
    t_pitch = (pp - pm) / (2 * cal_tilt)
    return (r0, t_roll, t_pitch,
            max(float((t_roll ** 2).sum()), 1e-9),
            max(float((t_pitch ** 2).sum()), 1e-9))
