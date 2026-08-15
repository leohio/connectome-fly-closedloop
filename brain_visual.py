#!/usr/bin/env python
"""統合22: FlyWire視覚サブ回路 (オセリ+VS/HS → 中継 → DN) のLIF。

経路 (FlyWire 783実測):
  オセリ視細胞273 →(2543syn)→ OCG20 →(851syn)→ 標的DN23 (直接!)
  VS/HS38 + OCG → 中継123 →(7101syn)→ DN
標的DN型 = MANCで翅操舵MNへ直接結合する上位型
  (DNa10/a08/a04/a05, DNp26/p03/p18, DNb01/b04, DNg04)

符号化 (水平線センサ):
  オセリ視細胞: r = R0·(1 + K·tilt) — 左右側はロール反対称、中央/両側共通はピッチ
  VS/HS: 回転レート符号化 (ドリフトでは弱い、含めるだけ)

mode=tuning : ロール/ピッチ傾き → DN発火率の伝達関数を測定
"""
import sys
import numpy as np
import pandas as pd
from brian2 import (NeuronGroup, Synapses, PoissonGroup, SpikeMonitor,
                    Network, mV, ms, Hz, prefs)

prefs.codegen.target = "numpy"

BRAIN = "../fly-brain-sim/Drosophila_brain_model"
DNT = ["DNa10", "DNa08", "DNa04", "DNa05", "DNp26", "DNp03", "DNb01",
       "DNb04", "DNg04", "DNp18"]
P = dict(v_0=-52, v_rst=-52, v_th=-45, t_mbr=20, tau=5, t_rfc=2.2,
         t_dly=1.8, w_syn=0.275, f_poi=250)
R0_OC = 150.0        # オセリ視細胞の基準レート
K_TILT = 4.0        # 傾き→レート変調 [1/rad]
R0_VS = 40.0
K_RATE = 0.0       # 回転レート→VS/HS変調 [s/rad]
W_MIN = 3


EXC_FAC = 3.0       # 中継・DNの入力重みスケール (切り出しで失う背景興奮の補償)
ITN_MV = 4.0        # 中継・DNの緊張性脱分極 [mV] (閾値7mV以下=自発発火なし)


def build_subcircuit(shuffle=None, shuffle_seed=0):
    """(neu, syn, ids, idx, meta) を返す。metaは各集団のroot_id配列。"""
    ct = pd.read_csv(f"{BRAIN}/consolidated_cell_types.csv")
    cl = pd.read_csv(f"{BRAIN}/classification.csv").set_index("root_id")
    conn = pd.read_parquet(
        f"{BRAIN}/Connectivity_783.parquet",
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity",
                 "Excitatory"])
    t = ct.primary_type.astype(str)
    ocr = ct[t == "ocellar_retinula_cell"].root_id.values
    ocg = ct[t.str.startswith("OCG")].root_id.values
    vshs = ct[t.str.match(r"VS\d|VSm|VST\d|HS[ENS]")].root_id.values
    dn = ct[t.isin(DNT)]
    dn_ids = dn.root_id.values
    src = set(ocr) | set(ocg) | set(vshs)
    c5 = conn[conn.Connectivity >= 5]
    down = c5[c5.Presynaptic_ID.isin(src)]
    up = c5[c5.Postsynaptic_ID.isin(set(dn_ids))]
    relay = np.array(sorted((set(down.Postsynaptic_ID)
                             & set(up.Presynaptic_ID))
                            - src - set(dn_ids)))
    ids = np.concatenate([ocr, ocg, vshs, relay, dn_ids])
    ids = pd.unique(ids)
    idset = set(ids)
    idx = {int(b): i for i, b in enumerate(ids)}
    e = conn[conn.Presynaptic_ID.isin(idset)
             & conn.Postsynaptic_ID.isin(idset)
             & (conn.Connectivity >= W_MIN)]
    pre = np.array([idx[int(b)] for b in e.Presynaptic_ID])
    post = np.array([idx[int(b)] for b in e.Postsynaptic_ID])
    w = e.Connectivity.values * np.where(e.Excitatory.values > 0, 1.0, -1.0)
    if shuffle == "all":
        rng = np.random.default_rng(shuffle_seed)
        post = rng.permutation(post)
    side = cl.reindex(ids)["side"].astype(str).values
    typ = ct.set_index("root_id").primary_type.reindex(ids).astype(str).values
    n = len(ids)
    ns = dict(v_0=P["v_0"]*mV, t_mbr=P["t_mbr"]*ms, tau=P["tau"]*ms,
              v_th=P["v_th"]*mV, v_rst=P["v_rst"]*mV)
    eqs = """
    dv/dt = (v_0 - v + g + itn) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau : volt (unless refractory)
    rfc : second
    itn : volt
    """
    neu = NeuronGroup(n, eqs, threshold="v > v_th",
                      reset="v = v_rst; g = 0*mV", refractory="rfc",
                      method="linear", namespace=ns, name="vis")
    neu.v = P["v_0"]*mV; neu.g = 0*mV; neu.rfc = P["t_rfc"]*ms
    neu.itn = 0*mV
    # 中継・DNの動作点補償 (サブ回路切り出しで失う背景入力のモデル)
    boost = np.array([idx[int(b)] for b in
                      np.concatenate([relay, dn_ids]) if int(b) in idx])
    fac = np.ones(n)
    fac[boost] = EXC_FAC
    for i in boost:
        neu.itn[i] = ITN_MV*mV
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                   delay=P["t_dly"]*ms, name="vsyn")
    syn.connect(i=pre, j=post)
    syn.w = w * P["w_syn"] * fac[post] * mV
    meta = dict(ocr=ocr, ocg=ocg, vshs=vshs, relay=relay, dn=dn,
                side=side, typ=typ)
    return neu, syn, ids, idx, meta


def sensory_attach(neu, idx, meta):
    """オセリ視細胞+VS/HSへ1対1 PoissonGroup を接続。"""
    sens = np.concatenate([meta["ocr"], meta["vshs"]])
    tgt = np.array([idx[int(b)] for b in sens])
    pg = PoissonGroup(len(sens), rates=0*Hz, name="vis_pg")
    sh = Synapses(pg, neu, on_pre="v_post += %f*mV" % (P["w_syn"]*P["f_poi"]),
                  name="vis_syn")
    sh.connect(i=np.arange(len(sens)), j=tgt)
    for i in tgt:
        neu.rfc[i] = 0*ms
    cl = pd.read_csv(f"{BRAIN}/classification.csv").set_index("root_id")
    sd = cl.reindex(sens)["side"].astype(str).values
    ssign = np.where(sd == "left", +1.0, np.where(sd == "right", -1.0, 0.0))
    is_oc = np.array([1.0]*len(meta["ocr"]) + [0.0]*len(meta["vshs"]))
    return pg, sh, ssign, is_oc


def vis_rates(roll, pitch, omr, omp, ssign, is_oc):
    """姿勢傾き(rad)と回転レート(rad/s) → 感覚レート"""
    oc = R0_OC*(1.0 + K_TILT*(ssign*roll + 0.6*pitch))
    vs = R0_VS*(1.0 + K_RATE*(ssign*omr + 0.5*omp))
    r = is_oc*oc + (1.0-is_oc)*vs
    return np.clip(r, 0, 300)


def dn_readout(meta, idx):
    """DNごとの (index, type, side) テーブル"""
    rows = []
    for _, r in meta["dn"].iterrows():
        rows.append((idx[int(r.root_id)], r.primary_type,
                     str(meta["side"][idx[int(r.root_id)]])))
    return rows


def mode_tuning():
    from brian2 import ms as _ms
    neu, syn, ids, idx, meta = build_subcircuit()
    print(f"サブ回路 {len(ids)} ニューロン, {len(syn.w[:])} シナプス", flush=True)
    pg, sh, ssign, is_oc = sensory_attach(neu, idx, meta)
    mon = SpikeMonitor(neu, record=True)
    net = Network(neu, syn, mon, pg, sh)
    dnrows = dn_readout(meta, idx)
    conds = [("水平", 0, 0), ("roll+0.3", 0.3, 0), ("roll-0.3", -0.3, 0),
             ("pitch+0.3", 0, 0.3), ("pitch-0.3", 0, -0.3)]
    T = 0.5
    rate = {}
    prev_cnt = np.zeros(len(ids))
    for name, ro, pi in conds:
        pg.rates = vis_rates(ro, pi, 0, 0, ssign, is_oc) * Hz
        net.run(T*1000*_ms)
        cnt = mon.count[:].copy()
        rate[name] = (cnt - prev_cnt) / T      # このブロックの発火率
        prev_cnt = cnt
    print(f"\n{'DN':10s}{'side':6s}{'基準Hz':>8s}"
          f"{'Δr+':>7s}{'Δr-':>7s}{'Δp+':>7s}{'Δp-':>7s}")
    for i, ty, sd in sorted(dnrows, key=lambda x: (x[1], x[2])):
        b = rate["水平"][i]
        d = [rate[k][i] - b for k in
             ["roll+0.3", "roll-0.3", "pitch+0.3", "pitch-0.3"]]
        print(f"{ty:10s}{sd:6s}{b:8.1f}{d[0]:+7.1f}{d[1]:+7.1f}"
              f"{d[2]:+7.1f}{d[3]:+7.1f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tuning"
    {"tuning": mode_tuning}[mode]()
