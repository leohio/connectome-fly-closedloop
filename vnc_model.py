#!/usr/bin/env python
"""MANC v1.0 腹髄全体の LIF モデル (Brian2)。

Shiu et al. 2024 の全脳モデルと同一の方程式・定数を MANC 実測配線に適用する。
- 23,188 traced ニューロン
- シナプス重み = シナプス数 × 0.275 mV × 符号(ACh:+, GABA/Glu:−)
- 刺激: 任意の bodyId 集合へのポアソン入力 (光遺伝学の モデル)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent.parent / "vnc-connectome" / "downloads"
PROPS = Path(__file__).parent / "neuron-properties.feather"
CACHE = Path(__file__).parent / "outputs" / "net_cache.npz"

MDN_BODYIDS = [13438, 13809, 14419, 14523]
W_MIN = 3  # シナプス数がこれ未満の結合は無視(速度のため)

# Shiu et al. 2024 定数
PARAMS = dict(v_0=-52, v_rst=-52, v_th=-45, t_mbr=20, tau=5, t_rfc=2.2,
              t_dly=1.8, w_syn=0.275, f_poi=250)


def build_arrays():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["ids"], d["pre"], d["post"], d["w"]
    neurons = pd.read_csv(DATA / "traced-neurons.csv")
    conns = pd.read_csv(DATA / "traced-connections.csv")
    props = pd.read_feather(PROPS)[["bodyId", "predictedNt"]]
    ids = neurons.bodyId.values
    idx = {b: i for i, b in enumerate(ids)}
    conns = conns[conns.weight >= W_MIN]
    sign_map = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}
    nt = props.set_index("bodyId").predictedNt.map(sign_map).fillna(1.0)
    sign = nt.reindex(conns.bodyId_pre).values
    pre = np.array([idx[b] for b in conns.bodyId_pre])
    post = np.array([idx[b] for b in conns.bodyId_post])
    w = conns.weight.values * sign  # 単位: シナプス数(符号付き)。mV換算はbrian2側
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, ids=ids, pre=pre, post=post, w=w)
    return ids, pre, post, w


def make_network(stim_bodyids, r_stim_hz=150, sensory_bodyids=None):
    """Brian2 ネットワークを構築して返す。

    sensory_bodyids を渡すと、その各ニューロンに1対1のPoissonGroupを接続して
    返す(戻り値5番目)。pg.rates を外から書き換えることで時変の感覚入力を
    注入できる(閉ループ用)。
    """
    from brian2 import (NeuronGroup, Synapses, PoissonInput, PoissonGroup,
                        SpikeMonitor, Network, mV, ms, Hz)
    ids, pre, post, w = build_arrays()
    idx = {b: i for i, b in enumerate(ids)}
    n = len(ids)
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau : volt (unless refractory)
    rfc : second
    """
    ns = dict(v_0=PARAMS["v_0"] * mV, t_mbr=PARAMS["t_mbr"] * ms,
              tau=PARAMS["tau"] * ms, v_th=PARAMS["v_th"] * mV,
              v_rst=PARAMS["v_rst"] * mV)
    neu = NeuronGroup(n, eqs, threshold="v > v_th",
                      reset="v = v_rst; g = 0*mV", refractory="rfc",
                      method="linear", namespace=ns, name="vnc")
    neu.v = PARAMS["v_0"] * mV
    neu.g = 0 * mV
    neu.rfc = PARAMS["t_rfc"] * ms
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                   delay=PARAMS["t_dly"] * ms, namespace=ns, name="syn")
    syn.connect(i=pre, j=post)
    syn.w = w * PARAMS["w_syn"] * mV
    pois = []
    for b in stim_bodyids:
        i = idx[b]
        pois.append(PoissonInput(target=neu[i:i + 1], target_var="v", N=1,
                                 rate=r_stim_hz * Hz,
                                 weight=PARAMS["w_syn"] * PARAMS["f_poi"] * mV))
        neu.rfc[i] = 0 * ms
    mon = SpikeMonitor(neu, record=True)
    objs = [neu, syn, mon, *pois]
    pg = None
    if sensory_bodyids is not None:
        sn_idx = np.array([idx[b] for b in sensory_bodyids if b in idx])
        pg = PoissonGroup(len(sn_idx), rates=0 * Hz, name="sensory_pg")
        syn_s = Synapses(pg, neu, on_pre="v_post += %f*mV" %
                         (PARAMS["w_syn"] * PARAMS["f_poi"]), name="syn_sensory")
        syn_s.connect(i=np.arange(len(sn_idx)), j=sn_idx)
        neu.rfc[sn_idx] = 0 * ms
        objs += [pg, syn_s]
    net = Network(*objs)
    if pg is not None:
        return net, mon, ids, idx, pg
    return net, mon, ids, idx


def leg_mn_table():
    s6 = pd.read_csv(DATA / "elife-96084-supp6-v1.csv", encoding="latin1")
    return s6[["bodyid", "soma_neuromere", "soma_side", "target"]]


if __name__ == "__main__":
    from brian2 import ms, prefs
    prefs.codegen.target = "numpy"
    import time
    t0 = time.time()
    net, mon, ids, idx = make_network(MDN_BODYIDS, r_stim_hz=150)
    print(f"build: {time.time()-t0:.1f}s")
    t0 = time.time()
    net.run(500 * ms)
    print(f"run 500ms: {time.time()-t0:.1f}s, total spikes: {mon.num_spikes}")

    trains = mon.spike_trains()
    mns = leg_mn_table()
    rates = []
    for _, r in mns.iterrows():
        i = idx.get(r.bodyid)
        if i is None:
            continue
        rates.append((r.soma_neuromere, r.soma_side, r.target,
                      len(trains[i]) / 0.5))
    df = pd.DataFrame(rates, columns=["seg", "side", "target", "hz"])
    print("\n=== MDN 150Hz 刺激時の脚MN平均発火率 [Hz] ===")
    print(df.groupby("seg").hz.mean().round(1).to_string())
    print("\n上位ターゲット筋:")
    print(df.groupby(["seg", "target"]).hz.mean().sort_values(ascending=False)
          .head(8).round(1).to_string())
    active = (df.hz > 1).sum()
    print(f"\n active leg MNs (>1Hz): {active}/{len(df)}")
