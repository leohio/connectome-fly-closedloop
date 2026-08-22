#!/usr/bin/env python
"""統合38: flybodyでの歩行の土台 — 接触床と立位。

flybodyの床 (vision_floor) は飛行用に contype=0 で追加されていた。歩行では
接触を有効化するが、MuJoCo 3.x は body単位の接触マスク (body_contype) を
コンパイル時にキャッシュするため、geom_contype の実行時書き換えだけでは
ブロードフェーズに乗らない — body側マスクも併せて更新する必要がある
(これを見逃すと mj_geomDistance は貫通を返すのに接触が生成されない)。

立位: キーフレーム0の脚姿勢は、脚関節の受動剛性だけで直立度+1.00の立位を
2秒以上保持できる (z≈0.127、接触10点)。
"""
import numpy as np
import mujoco
import openloop_hover as OH

LEG_KEYS = ["coxa", "femur", "tibia", "tarsus"]


def env():
    """接触床つきのflybodyと、脚アクチュエータ・立位姿勢を返す"""
    E = OH.env()
    m = E["m"]
    fid = next(i for i in range(m.ngeom)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
               == "vision_floor")
    m.geom_contype[fid] = 1
    m.geom_conaffinity[fid] = 1
    m.body_contype[0] |= 1
    m.body_conaffinity[0] |= 1
    d0 = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d0, 0)
    leg_act, leg_q, leg_names = [], [], []
    for i in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if nm and any(k in nm for k in LEG_KEYS):
            jid = m.actuator_trnid[i][0]
            leg_act.append(i)
            leg_q.append(float(d0.qpos[m.jnt_qposadr[jid]]))
            leg_names.append(nm)
    return dict(m=m, floor=fid, leg_act=np.array(leg_act),
                stance_q=np.array(leg_q), leg_names=leg_names,
                Q0=E["Q0"], ZT_W=E["ZT_W"], aid=E["aid"])


def reset_standing(m, d, z=0.14):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[2] = z
    mujoco.mj_forward(m, d)


def settle(m, d, W, T=0.5):
    """立位姿勢を保持して静定させる"""
    for _ in range(int(T / m.opt.timestep)):
        d.ctrl[:] = 0
        d.ctrl[W["leg_act"]] = W["stance_q"]
        mujoco.mj_step(m, d)


if __name__ == "__main__":
    W = env()
    m = W["m"]
    d = mujoco.MjData(m)
    reset_standing(m, d)
    settle(m, d, W, T=2.0)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    print(f"立位: z={d.qpos[2]:+.4f} 直立度={R[8]:+.2f} 接触={d.ncon}")
