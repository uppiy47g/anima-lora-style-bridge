# Changelog

このプロジェクトの主な変更を記録します。

## [3.1.0] - 2026-09-15

### Added

- 決定的な深さ対応と再利用可能なProcrustes / CCA特徴橋を組み合わせる
  `fit-v3-hybrid-bridge`
- v3の局所差分抽出・低ランク蒸留・validation RMS校正を維持した
  `distill-v3-hybrid`

## [3.0.0] - 2026-09-11

### Added

- 線形CKAと単調制約によるレイヤー対応
- ridge回帰によるベース特徴空間の整列
- LoRAあり・なしの活性差分を使うFitNets型蒸留
- SVD低ランク化後の出力RMSスケール校正
- `distill-v3` CLIと活性NPZ形式
- v2 ComfyUI Captureと元SDXL LoRAから局所差分を再構成する
  `prepare-v3-activations` CLI

### Compatibility

- v1の`convert`、`fit-calibration`、`validate`コマンドを維持

## [1.0.0] - 2026-09-03

### Added

- SD 1.5 / SDXL Transformer LoRAからAnimaの全28ブロックへの変換
- ソース深度に基づくブロック補間
- 複数差分の重み付き合成と指定ランクへの圧縮
- Procrustes / CCAキャリブレーション
- `float16`、`float32`、`bfloat16`出力
- 変換済みファイルの構造検証

### Safety

- 入力ファイルと同じパスへの出力を拒否
- 不正なランクと正則化係数をCLIで拒否
