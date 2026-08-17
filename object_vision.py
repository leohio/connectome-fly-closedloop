#!/usr/bin/env python
"""統合26 LoopB/C: 木のある世界と複眼→コネクトーム物体視。

LoopB: シーンに衝突可能な暗い木 (円柱) を追加し、左右複眼カメラ (48x48,
体固定・側前方視) でレンダリング。
LoopC: FlyWire実座標による本物のレチノトピー —
  光受容体R細胞11,151本の細胞体は眼表面に物理配置されている。
  PCAで眼面2軸を推定し、各R細胞→カメラ画素を対応づけ、画素輝度で駆動。
  実配線 (R→ラミナ→...→LC4/LPLC2/LC11→DNp01巨大繊維) を通った
  応答から物体方位 (LC左右差) と接近 (LPLC2/DNp01) を読み出す。
"""
import os
import numpy as np
import mujoco

EYE_RES = 48
EYE_DIRS = dict(L=[-0.8, -0.45, 0.1], R=[-0.8, 0.45, 0.1])
# 注: flybodyは頭が体-x方向。左眼=進行方向左は体-y側

_TREE_MODEL = {}


def get_tree_model(tree=(-18.0, 8.0), radius=1.2, height=24.0):
    """木 (衝突可能な暗い円柱) 入りの視覚シーン。"""
    key = (tree, radius, height)
    if key in _TREE_MODEL:
        return _TREE_MODEL[key]
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
        xml = xml.replace('<mujoco model=', '<mujoco model=', 1)
        if '<worldbody>' in xml and '<size ' not in xml:
            xml = xml.replace('<worldbody>',
                '<size memory="32M"/><worldbody>', 1)
        if '<asset>' in xml:
            xml = xml.replace('<asset>',
                '<asset><texture type="skybox" builtin="gradient" '
                'rgb1="1 1 1" rgb2="0.85 0.9 1" width="64" height="64"/>', 1)
        xml = xml.replace(
            '<worldbody>',
            '<worldbody>'
            '<geom name="vision_floor" type="plane" size="120 120 0.1" '
            'pos="0 0 0" rgba="0.9 0.9 0.9 1" contype="1" conaffinity="1"/>'
            f'<geom name="tree" type="cylinder" size="{radius} {height/2}" '
            f'pos="{tree[0]} {tree[1]} {height/2}" rgba="0.05 0.08 0.02 1" '
            'contype="1" conaffinity="1"/>', 1)
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
    m.vis.global_.fovy = 90.0     # 広視野複眼
    _TREE_MODEL[key] = m
    return m


class CompoundEye:
    """左右複眼カメラ: 体固定方向をレンダリングし輝度画像を返す"""

    def __init__(self, m, res=EYE_RES):
        self.m = m
        self.res = res
        self.ren = mujoco.Renderer(m, res, res)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def images(self, d):
        out = {}
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, d.qpos[3:7])
        Rb = Rm.reshape(3, 3)
        for k, v in EYE_DIRS.items():
            dw = Rb @ np.array(v)
            dw /= np.linalg.norm(dw)
            self.cam.lookat[:] = d.qpos[:3] + dw
            self.cam.distance = 1.0
            self.cam.azimuth = np.degrees(np.arctan2(dw[1], dw[0])) + 180.0
            self.cam.elevation = -np.degrees(
                np.arcsin(np.clip(dw[2], -1, 1)))
            self.ren.update_scene(d, camera=self.cam)
            out[k] = self.ren.render().mean(axis=2) / 255.0   # [0,1]輝度
        return out


def mode_scene_test():
    """LoopB検証: 木は見えるか、方位と距離は画像に現れるか"""
    import phase_reflex as PR
    env = PR._fly_env()
    Q0 = env["Q0"]
    m = get_tree_model(tree=(-18.0, 8.0))
    d = mujoco.MjData(m)
    eye = CompoundEye(m)

    def look(pos, yaw):
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = pos
        q = np.array(Q0)
        dq = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, dq, q)
        d.qpos[3:7] = out
        mujoco.mj_forward(m, d)
        im = eye.images(d)
        dkL = float((im["L"] < 0.35).mean())
        dkR = float((im["R"] < 0.35).mean())
        return dkL, dkR

    tree = np.array([-18.0, 8.0])
    for pos, label in [((0, 0, 12), "原点"), ((-8, 4, 12), "中間"),
                       ((-14, 6.5, 12), "近距離")]:
        bearing = np.arctan2(tree[1] - pos[1], tree[0] - pos[0])
        for yaw_off, yl in [(0.0, "正面"), (0.5, "左28°"), (-0.5, "右28°"),
                            (np.pi, "背面=基線")]:
            dkL, dkR = look(pos, bearing + yaw_off)
            print(f"{label} 木{yl}: 暗画素率 L={dkL:.3f} R={dkR:.3f} "
                  f"(L-R={dkL-dkR:+.3f})", flush=True)


if __name__ == "__main__":
    mode_scene_test()


# ============ LoopC: R細胞レチノトピー + 実配線物体視回路 ============
import pandas as pd

BRAIN = "../fly-brain-sim/Drosophila_brain_model"
NET_CACHE = "outputs/objnet_cache.npz"
R0_PHOT = 120.0     # 光受容体基準レート (明所)
KEEP_H1 = 4000      # ラミナ層の保持数 (R入力強度上位)
KEEP_BR = 4000      # ブリッジ層の保持数 (経路強度上位)
W_MIN_OV = 5


def build_object_arrays():
    """刈り込んだ物体視サブ回路 (R→hop1→bridge→LC→DNp01) の配列を構築"""
    if os.path.exists(NET_CACHE):
        z = np.load(NET_CACHE, allow_pickle=True)
        return (z["ids"], z["pre"], z["post"], z["w"], z["r_ids"],
                z["r_uv"], z["r_side"], dict(z["groups"].item()))
    ct = pd.read_csv(f"{BRAIN}/consolidated_cell_types.csv")
    cl = pd.read_csv(f"{BRAIN}/classification.csv").set_index("root_id")
    conn = pd.read_parquet(
        f"{BRAIN}/Connectivity_783.parquet",
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity",
                 "Excitatory"])
    t = ct.primary_type.astype(str)
    Rset = set(ct[t.str.match(r"^R1-6$|^R7$|^R8$")].root_id)
    lc_types = {"LC4": r"^LC4$", "LPLC2": r"^LPLC2$", "LC11": r"^LC11$",
                "LC6": r"^LC6$"}
    LCall = {}
    for nm, pat in lc_types.items():
        LCall[nm] = set(ct[t.str.match(pat)].root_id)
    LCu = set().union(*LCall.values())
    DN = set(ct[t.str.match(r"^DNp01$")].root_id)
    c = conn[conn.Connectivity >= W_MIN_OV]
    # hop1: Rの直接下流を入力強度で選抜
    h1 = c[c.Presynaptic_ID.isin(Rset) & ~c.Postsynaptic_ID.isin(Rset)]
    h1s = h1.groupby("Postsynaptic_ID").Connectivity.sum() \
        .sort_values(ascending=False)
    hop1 = set(h1s.head(KEEP_H1).index)
    # bridge: hop1下流 ∩ LC上流を経路強度で選抜
    h2 = c[c.Presynaptic_ID.isin(hop1)]
    into = h2.groupby("Postsynaptic_ID").Connectivity.sum()
    lcin = c[c.Postsynaptic_ID.isin(LCu)]
    outof = lcin.groupby("Presynaptic_ID").Connectivity.sum()
    cand = set(into.index) & set(outof.index) - Rset - hop1 - LCu - DN
    score = {b: float(into[b]) * float(outof[b]) for b in cand}
    bridge = set(sorted(cand, key=lambda b: -score[b])[:KEEP_BR])
    ids = np.array(sorted(Rset | hop1 | bridge | LCu | DN))
    idset = set(ids)
    idx = {int(b): i for i, b in enumerate(ids)}
    e = c[c.Presynaptic_ID.isin(idset) & c.Postsynaptic_ID.isin(idset)]
    pre = np.array([idx[int(b)] for b in e.Presynaptic_ID])
    post = np.array([idx[int(b)] for b in e.Postsynaptic_ID])
    w = e.Connectivity.values * np.where(e.Excitatory.values > 0, 1.0, -1.0)
    # レチノトピー: R細胞の実座標をPCAで眼面座標へ
    coords = pd.read_csv(f"{BRAIN}/coordinates.csv")
    coords = coords[coords.root_id.isin(Rset)].drop_duplicates("root_id")
    pos = np.array([[float(x) for x in p.strip("[]").split()]
                    for p in coords.position])
    side = cl.reindex(coords.root_id)["side"].astype(str).values
    r_ids, r_uv, r_side = [], [], []
    for s, sgn in [("left", 0), ("right", 1)]:
        msk = side == s
        P3 = pos[msk]
        rid = coords.root_id.values[msk]
        c0 = P3 - P3.mean(0)
        U, S, Vt = np.linalg.svd(c0, full_matrices=False)
        uv = c0 @ Vt[:2].T           # 眼面2軸
        uv = (uv - uv.min(0)) / (uv.max(0) - uv.min(0) + 1e-9)
        r_ids += [int(b) for b in rid]
        r_uv += list(uv)
        r_side += [sgn] * len(rid)
    r_ids = np.array(r_ids)
    r_uv = np.array(r_uv)
    r_side = np.array(r_side)
    groups = {nm: np.array([idx[int(b)] for b in v])
              for nm, v in LCall.items()}
    groups["DNp01"] = np.array([idx[int(b)] for b in DN])
    lc_side = {}
    for nm in list(lc_types) + ["DNp01"]:
        gg = groups[nm]
        sd = cl.reindex(ids[gg])["side"].astype(str).values
        lc_side[nm + "_L"] = gg[sd == "left"]
        lc_side[nm + "_R"] = gg[sd == "right"]
    groups.update(lc_side)
    np.savez_compressed(NET_CACHE, ids=ids, pre=pre, post=post, w=w,
                        r_ids=r_ids, r_uv=r_uv, r_side=r_side,
                        groups=np.array(groups, dtype=object))
    print(f"objnet: {len(ids)}細胞 {len(w)}エッジ (R={len(Rset)}, "
          f"hop1={len(hop1)}, bridge={len(bridge)}, LC={len(LCu)}, "
          f"DN={len(DN)})", flush=True)
    return ids, pre, post, w, r_ids, r_uv, r_side, groups


if __name__ == "__main__" and len(os.sys.argv) > 1 and \
        os.sys.argv[1] == "build":
    build_object_arrays()


P_LIF = dict(v_0=-52, v_rst=-52, v_th=-45, t_mbr=20, tau=5, t_rfc=2.2,
             t_dly=1.8, w_syn=0.275, f_poi=250)
EXC_OV = 3.0
ITN_OV = 4.0


def build_object_net():
    """LIF化した物体視回路。(net, mon, pg, mapping...) を返す"""
    from brian2 import (NeuronGroup, Synapses, PoissonGroup, SpikeMonitor,
                        Network, mV, ms, Hz)
    ids, pre, post, w, r_ids, r_uv, r_side, groups = build_object_arrays()
    idx = {int(b): i for i, b in enumerate(ids)}
    n = len(ids)
    ns = dict(v_0=P_LIF["v_0"]*mV, t_mbr=P_LIF["t_mbr"]*ms,
              tau=P_LIF["tau"]*ms, v_th=P_LIF["v_th"]*mV,
              v_rst=P_LIF["v_rst"]*mV)
    eqs = """
    dv/dt = (v_0 - v + g + itn) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau : volt (unless refractory)
    rfc : second
    itn : volt
    """
    neu = NeuronGroup(n, eqs, threshold="v > v_th",
                      reset="v = v_rst; g = 0*mV", refractory="rfc",
                      method="linear", namespace=ns, name="obj")
    neu.v = P_LIF["v_0"]*mV
    neu.g = 0*mV
    neu.rfc = P_LIF["t_rfc"]*ms
    neu.itn = 0*mV
    rset = set(int(b) for b in r_ids)
    deep = np.array([i for b, i in idx.items() if b not in rset])
    fac = np.ones(n)
    fac[deep] = EXC_OV
    for i in deep:
        neu.itn[i] = ITN_OV*mV
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                   delay=P_LIF["t_dly"]*ms, name="osyn")
    syn.connect(i=pre, j=post)
    syn.w = w * P_LIF["w_syn"] * fac[post] * mV
    # R細胞への1:1ポアソン駆動
    r_tgt = np.array([idx[int(b)] for b in r_ids])
    pg = PoissonGroup(len(r_ids), rates=0*Hz, name="phot_pg")
    sh = Synapses(pg, neu, on_pre="v_post += %f*mV" %
                  (P_LIF["w_syn"]*P_LIF["f_poi"]), name="photsyn")
    sh.connect(i=np.arange(len(r_ids)), j=r_tgt)
    for i in r_tgt:
        neu.rfc[i] = 0*ms
    mon = SpikeMonitor(neu, record=False)
    net = Network(neu, syn, mon, pg, sh)
    # 画素対応 (事前計算)
    pix = (np.clip(r_uv, 0, 0.999) * EYE_RES).astype(int)
    return net, mon, pg, pix, r_side, groups, n


def phot_rates(images, pix, r_side):
    """複眼画像 → 光受容体レート (明るいほど高発火 = 実物どおり)"""
    lumL = images["L"][pix[:, 1], pix[:, 0]]
    lumR = images["R"][pix[:, 1], pix[:, 0]]
    lum = np.where(r_side == 0, lumL, lumR)
    return R0_PHOT * lum


def mode_tuning():
    """LoopC検証: 木の方位・距離 → LC/DNp01応答"""
    from brian2 import Hz, ms as _ms
    import phase_reflex as PR
    env = PR._fly_env()
    Q0 = env["Q0"]
    m = get_tree_model(tree=(-18.0, 8.0))
    d = mujoco.MjData(m)
    eye = CompoundEye(m)
    net, mon, pg, pix, r_side, groups, n = build_object_net()
    print(f"LIF {n}細胞 起動", flush=True)
    tree = np.array([-18.0, 8.0])

    def pose(pos, yaw_off):
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = pos
        br = np.arctan2(tree[1]-pos[1], tree[0]-pos[0]) - yaw_off
        q = np.array(Q0)
        dq = np.array([np.cos(br/2), 0, 0, np.sin(br/2)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, dq, q)
        d.qpos[3:7] = out
        mujoco.mj_forward(m, d)
        return eye.images(d)

    conds = [("木なし(背)", (2, 0, 12), np.pi),
             ("遠20正面", (2, 0, 12), 0.0),
             ("中12左", (-7, 3.7, 12), 0.5),
             ("中12右", (-7, 3.7, 12), -0.5),
             ("近6正面", (-12.4, 5.9, 12), 0.0),
             ("極近3正面", (-15.2, 7, 12), 0.0)]
    prev = np.zeros(n)
    T = 0.3
    keys = ["LC4_L", "LC4_R", "LC11_L", "LC11_R", "LPLC2_L", "LPLC2_R",
            "DNp01_L", "DNp01_R"]
    print(f"{'条件':10s} " + " ".join(f"{k:>8s}" for k in keys), flush=True)
    for name, pos, off in conds:
        im = pose(pos, off)
        pg.rates = phot_rates(im, pix, r_side) * Hz
        net.run(T * 1000 * _ms)
        cnt = mon.count[:].copy()
        dr = (cnt - prev) / T
        prev = cnt
        row = [float(dr[groups[k]].mean()) if len(groups[k]) else -1
               for k in keys]
        print(f"{name:10s} " + " ".join(f"{v:8.1f}" for v in row), flush=True)
