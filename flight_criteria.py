#!/usr/bin/env python
"""安定飛行の基準 (統合39改訂2で明文化)。
ログ (t, z, up_world, floor_contact, mode) を受け取り合否を返す。
  up_world = 体の背側軸 (体z軸) の鉛直成分。+1=背中が真上、0=横倒し、-1=仰向け
S0 起動過渡の除外: 空中開始後1.0秒 (fly_engineの翅起動で直立が約0.1秒落ち、高度が
   約2沈んでから回復する過渡。実測: 姿勢は0.4秒、高度は0.8秒で回復) は S1/S2 から除外
S1 背中が上: 飛行区間の全時刻で up_world >= 0.5
S2 高度維持: 巡航区間で |z - z_cruise| <= 2.0、かつ飛行区間で床接触ゼロ
S3 持続:     巡航 >= 4.0 s
S4 着陸:     接地時 up_world >= 0.75、静定後 up_world >= 0.9
"""
import numpy as np


def evaluate(log, z_cruise, startup_skip=None):
    """startup_skip=(t_cut, dt): 空中開始直後 dt 秒の翅起動過渡を S1/S2 から除外する
    (fly_engineはfly()同様に起動後約0.1秒で直立が一時的に落ちてから回復する — 明示して除外)"""
    t = np.array([r[0] for r in log]); z = np.array([r[1] for r in log])
    up = np.array([r[2] for r in log]); con = np.array([r[3] for r in log])
    mode = np.array([r[4] for r in log])
    fly = np.isin(mode, ["takeoff", "flight", "descend"])
    if startup_skip is not None:
        fly = fly & (t >= startup_skip[0] + startup_skip[1])
    cru = (mode == "flight") & fly
    out = {}
    out["S1_背中が上"] = (bool(fly.any()) and float(up[fly].min()) >= 0.5,
                       f"飛行中の最小up={up[fly].min():.2f}" if fly.any() else "飛行なし")
    if cru.any():
        zerr = np.abs(z[cru] - z_cruise).max()
        out["S2_高度維持"] = (zerr <= 2.0 and not con[fly].any(),
                           f"巡航高度誤差max={zerr:.2f} 飛行中床接触={int(con[fly].sum())}")
        out["S3_持続"] = ((t[cru].max() - t[cru].min()) >= 4.0,
                        f"巡航{t[cru].max() - t[cru].min():.2f}s")
    else:
        out["S2_高度維持"] = (False, "巡航なし"); out["S3_持続"] = (False, "巡航なし")
    st = mode == "settle"
    if st.any():
        i0 = np.argmax(st)
        out["S4_着陸"] = (up[i0] >= 0.75 and up[st][-1] >= 0.9,
                        f"接地up={up[i0]:.2f} 静定後up={up[st][-1]:.2f}")
    else:
        out["S4_着陸"] = (False, "着陸なし")
    ok = all(v[0] for v in out.values())
    return ok, out


def report(ok, out):
    for k, (p, msg) in out.items():
        print(f"  [{'合格' if p else '不合格'}] {k}: {msg}")
    print("  総合:", "安定飛行 成立" if ok else "不成立")
