# Anima LoRA Style Bridge

SD 1.5 / SDXL向けLoRAの画風を、Anima LoRAとして再利用するための
実験的な重み橋渡しツールです。

> [!WARNING]
> このツールは異なるモデル構造間で重みを近似移植します。元LoRAの見た目や
> 挙動を完全に再現するものではありません。変換結果は必ず手元で確認してください。

## 特徴

- ソース側のTransformerブロックをAnimaの全28ブロックへ線形補間
- 同じ出力層へ集約される低ランク差分を、上書きせず重み付きで合成
- 合成後の差分を指定ランクへ圧縮し、Frobeniusノルムを保持
- `float16`、`float32`、`bfloat16`出力
- ペア活性値から求めたProcrustes / CCA射影を任意で利用可能（動作未確認）

## 必要環境

- Python 3.9以降
- PyTorch
- NumPy
- safetensors（Pythonライブラリ）

## インストール

```console
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install .
```

開発用ツールも含める場合:

```console
python -m pip install -e ".[dev]"
```

## 使い方

### 基本変換

```console
anima-style-bridge convert input.safetensors -o output.safetensors
```

`convert`は省略できます。

```console
anima-style-bridge input.safetensors
```

出力先を省略すると、入力ファイルと同じ場所ではなく現在のディレクトリに
`<入力名>_anima_style.safetensors`を作成します。

主なオプション:

```text
--rank N            出力ランク（既定値: 16）
--dtype TYPE        float16 / float32 / bfloat16（既定値: float16）
--calibration FILE  学習済み射影を格納したNPZ
```

### 出力の検証

```console
anima-style-bridge validate output.safetensors
```

LoRAペアの有無、ランク、Anima層の入出力次元を検査します。

### 射影のキャリブレーション

> [!CAUTION]
> Procrustes / CCA射影を利用するキャリブレーション機能は、実モデルでは
> 動作未確認です。

同じサンプルに対するソースモデルとターゲットモデルの活性値をNPZへ保存します。

```text
<name>.source = [samples, source_dim]
<name>.target = [samples, target_dim]
```

変換時には、次の順で射影名を検索します。

```text
<layer>.<side>.projection
<side>.<source_dim>x<target_dim>.projection
```

`side`は`in`または`out`です。例:

```text
self_attn.q_proj.in.projection
in.320x2048.projection
```

射影を作成して変換に使う例:

```console
anima-style-bridge fit-calibration activations.npz -o anima_alignment.npz
anima-style-bridge convert input.safetensors --calibration anima_alignment.npz
```

`--method`には`procrustes`（既定値）または`cca`を指定できます。
キャリブレーションを省略した場合は、決定的な線形リサンプリングを使用します。

## 対応する入力キー

Diffusers形式とSGM形式のSD 1.5 / SDXL Transformer LoRAキーを扱います。
AttentionのQ/K/V/出力射影と、Feed Forward層を変換対象とします。
未対応のキーはスキップされ、処理後に件数を表示します。

## 開発

```console
python -m unittest discover -s tests -v
```

## セキュリティと配布

モデルファイル、キャリブレーションデータ、変換済みファイルはリポジトリへ
含めないでください。このリポジトリの`.gitignore`では、`*.safetensors`と
`*.npz`を既定で除外しています。

## ライセンス

[MIT License](LICENSE)
