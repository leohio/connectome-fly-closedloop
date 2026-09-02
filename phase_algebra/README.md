# phase_algebra

位相制御 + 線形代数による、ハエ飛行制御系の簡略化と解析。計画と結果は [STRATEGY.md](STRATEGY.md)、最小モデルは [MODEL.md](MODEL.md)。

実行はリポジトリ直下から (外部データを相対パスで読むため):

```bash
../fly-body-sim/venv/bin/python phase_algebra/ident_plant.py      # P1: 機体 F, G の同定
../fly-body-sim/venv/bin/python phase_algebra/closed_loop.py      # P2: 閉ループ行列と遅延の壁
../fly-body-sim/venv/bin/python phase_algebra/hover_growth.py     # P2 検証
```

| ファイル | 内容 |
|---|---|
| `ident_plant.py` | P1。ホバー動作点で微小摂動を与え、1羽ばたきのストロボ写像 F (9×9), G (9×5) を最小二乗で同定。固有値と線形予測の検証つき |
| `outputs/plant_FG.npz` | 同定結果 (F, G, c, 参照状態) |
| `closed_loop.py` | P2。身体・筋・感覚・制御の 4 行列を合成した閉ループ行列 T。遅延 d を振って速い姿勢モードの |λ| を予測 |
| `hover_growth.py`, `floquet_check.py` | P2 検証。同じ座標で非線形閉ループを回し生存・成長率を実測 |
| `wiring_operator.py` | P3。回路から配線配列・符号化・MN指標を抽出 (キャッシュ)、位相和代理 |
| `threshold_surrogate.py`, `wta_invariant.py`, `encoding_coherence.py` | P3。閾値交差代理 / 勝者総取り不変量 / 符号化コヒーレンス不変量 |
| `circle_map.py` | P3。円周写像 (位相同期ループ) 代理 — 実配線の S と復号誤差を構造だけから再現 |
| `floquet_cycle.py` | P2 補遺。整数ビート断面の Floquet 有限差分 (失敗として記録) |
| `walk_phase.py` | P4。脚位相の複素コヒーレンス行列と Kuramoto 結合フィット (結果: 歩容が振動子系として不成立、`outputs/walk_phase_r{2,3}.json`) |
| `vnc_rhythm.py` | P4b。身体なし腹髄単独の脚 MN 集団リズム解析 (結果: ポアソン雑音と同等、`outputs/vnc_rhythm_*.json`, `vnc_rhythm_null.json`) |
| `MODEL.md` | P5。全体を 1 ページの最小複素線形モデルにまとめた草稿 |
| `closed_loop.py` | P2。身体・筋・感覚 (遅れ+遅延)・制御則を合成した閉ループ行列 T。速い姿勢モードと遅いドリフトモードを分けて遅延 d ごとに |λ| を出す |
| `hover_growth.py` | P2 検証。同じ座標で非線形閉ループを回し、遅延ごとの生存と成長率を測る |
| `floquet_check.py` | P2 検証。摂動あり/なしの2軌道差から小摂動の成長率を測る (リミットサイクルの発見に至った) |
| `outputs/closed_loop.json`, `hover_growth.json`, `floquet_check.json` | 上記の結果 |
