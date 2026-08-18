#!/usr/bin/env python
"""対照: 「翅の変調を一切かけない開ループ」を厳密に測る。

統合28-30で載せてきた変調 (蒸留FF・ハルテア反射・較正復号) が、
そもそも「何もしない」より良いのかを確認する基準線。
"""
import numpy as np
from multiprocessing import Pool
import openloop_hover as OH


def job(s):
    return OH.rollout(np.zeros(12), pert_seed=s)


if __name__ == "__main__":
    with Pool(4) as p:
        r = p.map(job, range(4))
    sv = np.array([x[0] for x in r])
    up = np.array([x[1] for x in r])
    print(f"開ループ(変調なし, hover_params): 生存 平均{sv.mean():.2f}s "
          f"± {sv.std():.2f} 直立{up.mean():+.2f} "
          f"(各 {', '.join(f'{x:.2f}' for x in sv)})", flush=True)
    print("既測: 教師+遅いω 0.25s / 実回路閉ループ 0.44s / 実回路FF 0.38s "
          "/ 教師+全帯域ω 3.00s", flush=True)
    print("DONE", flush=True)
