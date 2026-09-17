# Anima LoRA Style Bridge v3.1

SDXL向けLoRAが生じさせる活性差分を、通常のAnima LoRAへ蒸留するための
実験的な変換ツールです。従来の重み近似変換も引き続き利用できます。

> [!WARNING]
> このツールは異なるモデル構造間で重みを近似移植します。元LoRAの見た目や
> 挙動を完全に再現するものではありません。変換結果は必ず手元で確認してください。

## 特徴

- 線形CKAと単調対応制約によるSDXL・Anima間の層対応
- ridge回帰によるベース特徴空間の整列
- SDXL LoRAの`styled - base`活性差分をAnima LoRAへ低ランク蒸留
- 低ランク化後の出力RMS校正
- `float16`、`float32`、`bfloat16`出力
- v1の深度補間およびProcrustes / CCA重み射影も利用可能

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

ComfyUIでCaptureノードを使う場合は、このリポジトリをComfyUIの
`custom_nodes/anima_lora_style_bridge`へ配置し、ComfyUI環境で依存関係を
インストールしてから再起動してください。

```console
cd ComfyUI\custom_nodes
git clone https://github.com/uppiy47g/anima-lora-style-bridge.git anima_lora_style_bridge
cd anima_lora_style_bridge
python -m pip install .
```

## 使い方

### v3: 活性差分蒸留

#### Activation CaptureからNPZを作る

同梱のv2互換`Anima Activation Capture Start/Finish`を使い、
`aggregation=mean`で次の2ファイルを取得します。

1. 元SDXL LoRAを適用したSDXLを`source_lora`としてCapture
2. LoRAなしAnima-Baseを`target_base`としてCapture

SDXL側のComfyUI LoRA loaderには、変換時に指定するものと同じLoRAファイルを
1つだけ適用してください。モデル強度も記録します。CLIP強度はデノイザーの局所
差分再構成には使われません。

```powershell
anima-style-bridge prepare-v3-activations `
  source_lora.npz `
  target_base.npz `
  original_sdxl_lora.safetensors `
  -o activations_v3.npz `
  --lora-strength 1.0
```

PowerShellで複数行に分ける場合、行継続にはバッククォート`` ` ``を使います。

このコマンドはCaptureされたSDXL層入力と元LoRAの重みから局所差分を計算し、
Capture出力から差し引いて同一入力に対するベース出力を復元します。その後、
SDXLとAnimaのモジュール名を以下のv3形式へ正規化します。

`--lora-strength`には、Capture時にComfyUIのLoRA loaderで指定した
`strength_model`と同じ値を指定してください。`tokens`集約はモデル間で行数が
一致しないため拒否されます。Diffusers形式のLoRAは、対象レイヤーをすべてCapture
した場合に限り深さ順で対応付けます。部分LoRAでは正確な対応が曖昧になるため、
SGM形式のLoRAを使うか、Capture対象をLoRA対象層だけに限定してください。

片方のCaptureだけに余分な実行を**末尾へ追記したことが分かっている場合**は、
共通する先頭行だけを使用できます。

```console
anima-style-bridge prepare-v3-activations ... --truncate-unpaired-tail
```

途中で実行が欠落した場合は、それ以降の行対応がずれるため、このオプションを
使わず、対応するCaptureを取り直してください。

同じサンプルから取得したSDXLとAnimaの中間活性をNPZへ保存します。キー形式は
次の通りです。

```text
source.<block>.<layer>.base
source.<block>.<layer>.styled
target.<block>.<layer>.base
target.<block>.<layer>.input
```

- `source.*.base`: SDXL線形層本体の出力
- `source.*.styled`: **同一の層入力**にLoRA分岐を加えたSDXL線形層の出力
- `target.*.base`: LoRAなしAnima線形層の出力
- `target.*.input`: 同じAnima線形層に入る正規化後の入力
- 配列形状: `[samples, features]`または`[..., features]`
- flatten後の行数は全配列で同一にし、行の対応関係を保つ
- `source.<block>`はSDXL内の深さ順に0から番号を付ける
- `target.<block>`はAnimaのブロック番号0～27を使用する

対応レイヤー名:

```text
self_attn.q_proj        self_attn.k_proj
self_attn.v_proj        self_attn.output_proj
cross_attn.q_proj       cross_attn.k_proj
cross_attn.v_proj       cross_attn.output_proj
mlp.layer1              mlp.layer2
```

各配列の行は意味的に対応している必要があります。同じ画像を各モデルのVAEで
符号化し、対応するノイズ水準と空間位置から取得してください。単に同じseedと
step番号を使っただけの活性は、潜在空間とノイズスケジュールが異なるため、
対応サンプルとは見なせません。
解像度が異なる層では、正規化座標から同数の位置を層別にサンプリングして
ください。プロンプト、ノイズ水準、空間位置が偏らないよう行を構成します。

`source.*.base`と`source.*.styled`を別々の完全な推論から取得すると、上流層の
変化まで各層の差分に重複して入り、出力LoRAが過剰に効きます。LoRAあり推論中の
各層入力を固定したまま、元の線形演算とLoRA加算後の出力を同じフック内で記録
してください。

```powershell
anima-style-bridge distill-v3 activations_v3.npz `
  -o style_anima_v3.safetensors `
  --rank 16 `
  --alignment-regularization 1e-3 `
  --distillation-regularization 1e-3 `
  --min-cka 0.05 `
  --max-scale 10 `
  --max-nrmse 1.0 `
  --validation-fraction 0.2 `
  --seed 0
```

処理はレイヤー種別ごとにCKA行列を計算し、SDXLの深さ順を逆転させない対応を
選びます。その後、ベース活性間の特徴写像を学習し、写像した教師差分を再現する
Anima LoRAを閉形式ridge回帰とSVDで生成します。最後に、低ランク化後の出力RMSを
教師目標に合わせます。整列用写像は出力ファイルには含まれないため、推論時には
Animaと生成されたLoRAだけを使用します。

`--min-cka`未指定時は変換を拒否しませんが、CKAが0.1未満の対応には警告を表示
します。最初の実験では閾値なしで分布を確認し、その後データに合わせて閾値を
設定してください。各層には低ランク近似後の`nrmse`も表示します。
`--max-scale`を超える校正が必要な層は、活性不足またはランク不足として拒否します。
同様に、`--max-nrmse`を超えて教師差分の方向を再現できない層も拒否します。
CKAとridge回帰には学習行だけを使い、RMS校正とNRMSEには
`--validation-fraction`で確保した行を使用します。分割は`--seed`で再現できます。

### v3.1: ハイブリッド軽量蒸留

ハイブリッド方式は、v3の同一入力に対する局所LoRA差分抽出とvalidation RMS校正を
維持し、CKA層探索を決定的な深さ対応へ、ridge特徴整列を再利用可能な
Procrustes / CCA橋へ置き換えます。

最初に、SDXLとAnimaの組み合わせに対する橋を学習します。

```powershell
anima-style-bridge fit-v3-hybrid-bridge `
  "activations_v3.npz" `
  -o "sdxl_to_anima_hybrid_bridge.npz" `
  --method procrustes
```

続いて、同じv3活性からAnima LoRAを蒸留します。

```powershell
anima-style-bridge distill-v3-hybrid `
  "activations_v3.npz" `
  --bridge "sdxl_to_anima_hybrid_bridge.npz" `
  -o "style_anima_v3_hybrid.safetensors" `
  --rank 16 `
  --distillation-regularization 1e-3 `
  --max-scale 10 `
  --max-nrmse 1.0 `
  --validation-fraction 0.2 `
  --seed 0
```

`procrustes`は高速で安定し、少量データ向けの既定値です。`cca`は活性の共分散も
補正しますが、特に2048次元以上の層で計算時間とメモリ使用量が増えます。

橋はレイヤー種別とソース・ターゲット次元ごとに共有され、深さ対応は各レイヤーの
先頭・末尾を一致させて線形に割り当てます。同じSDXL・Animaのモデル版、
Capture対象、活性次元であれば、橋を別LoRAの変換へ再利用できます。ただし、
プロンプト分布やモデル版が大きく異なる場合は橋を再学習してください。

| 方式 | 層対応 | 特徴整列 | 推奨用途 |
|---|---|---|---|
| `distill-v3` | CKA単調対応 | 層ごとのridge | 十分なCaptureによる品質優先 |
| `distill-v3-hybrid` | 決定的な深さ対応 | 共有Procrustes / CCA橋 | 少量データ、多数LoRA、再現性優先 |

> [!IMPORTANT]
> CaptureノードはComfyUI上の中間活性を収集しますが、SDXLとAnimaの
> モデル読み込み、プロンプト、seed、サンプラー設定、実行順の対応は
> ワークフロー側で管理してください。

> [!NOTE]
> Anima公式の推奨に従い、蒸留先の活性取得にはAnima-Baseを使用してください。

### v1互換: 基本変換

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
