# InternLM

<div align="center">

<img src="./doc/imgs/logo.svg" width="200"/>
  <div> </div>
  <div align="center">
    <b><font size="5">InternLM</font></b>
    <sup>
      <a href="https://internlm.intern-ai.org.cn/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    <div> </div>
  </div>

[![license](./doc/imgs/license.svg)](./LICENSE)
[![evaluation](./doc/imgs/compass_support.svg)](https://github.com/internLM/OpenCompass/)
[![Documentation Status](https://readthedocs.org/projects/internlm/badge/?version=latest)](https://internlm.readthedocs.io/zh_CN/latest/?badge=latest)

[📘使用法](./doc/en/usage.md) |
[🛠️インストール](./doc/en/install.md) |
[📊トレーニングパフォーマンス](./doc/en/train_performance.md) |
[👀モデル](#model-zoo) |
[🆕更新ニュース](./CHANGE_LOG.md) |
[🤔Issues 報告](https://github.com/InternLM/InternLM/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>

## はじめに

InternLM は、70 億のパラメータを持つベースモデルと、実用的なシナリオに合わせたチャットモデルをオープンソース化しています。このモデルには以下の特徴があります:

- 何兆もの高品質なトークンをトレーニングに活用し、強力な知識ベースを確立します。
- 8k のコンテキストウィンドウ長をサポートし、より長い入力シーケンスと強力な推論機能を可能にする。
- ユーザが独自のワークフローを柔軟に構築できるよう、汎用性の高いツールセットを提供します。

さらに、大規模な依存関係を必要とせずにモデルの事前学習をサポートする軽量な学習フレームワークが提供されます。単一のコードベースで、数千の GPU を持つ大規模クラスタでの事前学習と、単一の GPU での微調整をサポートし、顕著な性能最適化を達成します。InternLM は、1024GPU でのトレーニングにおいて 90% 近いアクセラレーション効率を達成しています。

## 新闻

InternLM-7B-Chat v1.1 は、コード インタプリタと関数呼び出し機能を備えてリリースされました。 [Lagent](https://github.com/InternLM/lagent) で試すことができます。

## InternLM-7B

### パフォーマンス評価

オープンソースの評価ツール [OpenCompass](https://github.com/internLM/OpenCompass/) を用いて、InternLM の総合的な評価を行った。この評価では、分野別能力、言語能力、知識能力、推論能力、理解能力の 5 つの次元をカバーしました。以下は評価結果の一部であり、その他の評価結果については [OpenCompass leaderboard](https://opencompass.org.cn/rank) をご覧ください。

| データセット\モデル | **InternLM-Chat-7B** | **InternLM-7B** | LLaMA-7B | Baichuan-7B | ChatGLM2-6B | Alpaca-7B | Vicuna-7B |
| ---------------- | -------------------------- | --------------------- | -------- | ----------- | ----------- | --------- | --------- |
| C-Eval(Val)      | 53.2                       | 53.4                  | 24.2     | 42.7        | 50.9        | 28.9      | 31.2      |
| MMLU             | 50.8                       | 51.0                  | 35.2*    | 41.5        | 46.0        | 39.7      | 47.3      |
| AGIEval          | 42.5                       | 37.6                  | 20.8     | 24.6        | 39.0        | 24.1      | 26.4      |
| CommonSenseQA    | 75.2                       | 59.5                  | 65.0     | 58.8        | 60.0        | 68.7      | 66.7      |
| BUSTM            | 74.3                       | 50.6                  | 48.5     | 51.3        | 55.0        | 48.8      | 62.5      |
| CLUEWSC          | 78.6                       | 59.1                  | 50.3     | 52.8        | 59.8        | 50.3      | 52.2      |
| MATH             | 6.4                        | 7.1                   | 2.8      | 3.0         | 6.6         | 2.2       | 2.8       |
| GSM8K            | 34.5                       | 31.2                  | 10.1     | 9.7         | 29.2        | 6.0       | 15.3      |
| HumanEval        | 14.0                       | 10.4                  | 14.0     | 9.2         | 9.2         | 9.2       | 11.0      |
| RACE(High)       | 76.3                       | 57.4                  | 46.9*    | 28.1        | 66.3        | 40.7      | 54.0      |

- 評価結果は [OpenCompass 20230706](https://github.com/internLM/OpenCompass/) (*印のあるデータは原著論文からの引用を意味する)から取得したもので、評価設定は [OpenCompass](https://github.com/internLM/OpenCompass/) が提供する設定ファイルに記載されています。
- 評価データは、[OpenCompass](https://github.com/internLM/OpenCompass/) のバージョンアップにより数値的な差異が生じる可能性がありますので、[OpenCompass](https://github.com/internLM/OpenCompass/) の最新の評価結果をご参照ください。

### Model Zoo

InternLM 7B と InternLM 7B チャットは、InternLM を使って訓練され、オープンソース化されています。モデルの重みは 2 つのフォーマットで提供されています。Transformers フォーマットを使ってモデルをロードするだけでなく、InternLM を使って直接重みをロードして、さらに事前トレーニングや人間の好みアライメントトレーニングを行うこともできます。

| モデル                         | InternLM フォーマット Weight ダウンロードリンク                                                                                                                 | Transformers フォーマット Weight ダウンロードリンク                                         |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **InternLM 7B**         | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-7b)         | [🤗internlm/intern-7b](https://huggingface.co/internlm/internlm-7b)                 |
| **InternLM Chat 7B**    | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b)    | [🤗internlm/intern-chat-7b](https://huggingface.co/internlm/internlm-chat-7b)       |
| **InternLM Chat 7B 8k** | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b-8k) | [🤗internlm/intern-chat-7b-8k](https://huggingface.co/internlm/internlm-chat-7b-8k) |

**制限事項:** 学習過程におけるモデルの安全性を確保し、倫理的・法的要件に準拠したテキストを生成するようモデルに促す努力を行ってきたが、モデルのサイズと確率的生成パラダイムのため、モデルは依然として予期せぬ出力を生成する可能性がある。例えば、生成された回答には偏見や差別、その他の有害な内容が含まれている可能性があります。そのような内容を伝播しないでください。有害な情報の伝播によって生じるいかなる結果に対しても、私たちは責任を負いません。

### Transformers からのインポート

Transformers を使用して InternLM 7B チャットモデルをロードするには、以下のコードを使用します:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b-v1_1", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b-v1_1", trust_remote_code=True).cuda()
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "こんにちは", history=[])
>>> print(response)
こんにちは！どのようにお手伝いできますか？
>>> response, history = model.chat(tokenizer, "時間管理について3つの提案をお願いします", history=history)
>>> print(response)
もちろんです！以下に簡潔な形で時間管理に関する3つの提案を示します。

1. To-Doリストを作成し、優先順位を付ける: タスクを明確にリストアップし、それぞれの優先度を判断しましょう。重要で緊急なタスクから順に取り組むことで、効率的に作業を進めることができます。
2. 時間のブロック化を実践する: 作業を特定の時間枠に集中させるため、時間をブロック化しましょう。例えば、朝の2時間をメール対応に割り当て、午後の3時間をプロジェクトに集中するなど、タスクごとに時間を確保することが効果的です。
3. ディストラクションを排除する: 集中力を保つために、ディストラクションを最小限に抑えましょう。通知をオフにし、SNSやメールに気を取られないようにすることで、作業効率を向上させることができます。

これらの提案を実践することで、時間管理のスキルを向上させ、効果的に日々のタスクをこなしていくことができます。
```

### 対話

以下のコードを実行することで、フロントエンドインターフェースを通して InternLM Chat 7B モデルと対話することができます:

```bash
pip install streamlit==1.24.0
pip install transformers==4.30.2
streamlit run web_demo.py
```

その効果は以下の通り

![demo](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)

### デプロイ

[LMDeploy](https://github.com/InternLM/LMDeploy) を使って、InternLM をワンクリックでデプロイする。

1. まず、LMDeploy をインストールする:

```
  python3 -m pip install lmdeploy
```

2. クイックデプロイには以下のコマンドを使用します:

```
  python3 -m lmdeploy.serve.turbomind.deploy InternLM-7B /path/to/internlm-7b/model hf
```

3. モデルをエクスポートした後、以下のコマンドを使ってサーバーを起動し、デプロイされたモデルと会話することができます:

```
  python3 -m lmdeploy.serve.client {server_ip_addresss}:33337
```

[LMDeploy](https://github.com/InternLM/LMDeploy) は、InternLM をデプロイするための完全なワークフローを提供します。InternLM のデプロイの詳細については、[デプロイチュートリアル](https://github.com/InternLM/LMDeploy)を参照してください。

## ファインチューニングとトレーニング

### プリトレーニングとファインチューニングのチュートリアル

InternLMのインストール、データ処理、プレトレーニング、ファインチューニングを始めるには、[使用法チュートリアル](./doc/ja/usage.md)を参照してください。

### Transformers フォーマットへの変換

InternLM によって学習されたモデルは、コミュニティの様々なオープンソースプロジェクトとシームレスにドッキングするのに便利な Hugging Face Transformers 形式に簡単に変換することができます。`tools/convert2hf.py` の助けを借りて、トレーニング中に保存された weights は 1 つのコマンドで transformers 形式に変換することができます

```bash
python convert2hf.py --src_folder origin_ckpt/ --tgt_folder hf_ckpt/ --tokenizer tokenizes/tokenizer.model
```

変換後、以下のコードで transformers として読み込むことができます

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

## トレーニングシステム

### システムアーキテクチャ

詳細については、[システムアーキテクチャドキュメント](./doc/ja/structure.md) を参照してください。

### トレーニングパフォーマンス

InternLM は、Flash-Attention、Apex その他の高性能モデルオペレータを深く統合し、トレーニング効率を向上させます。Hybrid Zero 技術を構築することで、計算と通信の効率的なオーバーラップを実現し、トレーニング中のノード間の通信トラフィックを大幅に削減します。InternLM は 7B モデルを 8GPU から 1024GPU まで拡張することをサポートし、1000GPU スケールで最大 90% のアクセラレーション効率、180TFLOPS 以上のトレーニングスループット、GPU あたり平均 3600 トークン/秒以上を実現します。次の表は、異なる構成における InternLM のスケーラビリティテストデータです:

| GPU Number         | 8   | 16  | 32  | 64  | 128  | 256  | 512  | 1024  |
| ---------------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| TGS | 4078 | 3939 | 3919 | 3944 | 3928  | 3920  | 3835  | 3625   |
| TFLOPS  | 193 | 191  | 188  | 188  | 187   | 185   | 186   | 184    |

TGSは、GPUあたり1秒間に処理されるトークンの平均数を表します。パフォーマンステストデータの詳細については、[トレーニングパフォーマンスドキュメント](./doc/ja/train_performance.md)を参照してください。

## コントリビュート

我々は、InternLM を改善し、向上させるために尽力してくれたすべての貢献者に感謝している。コミュニティ・ユーザーのプロジェクトへの参加が強く推奨されます。プロジェクトへの貢献方法については、貢献ガイドラインを参照してください。

## 謝辞

InternLM コードベースは、上海 AI 研究所と様々な大学や企業の研究者によって貢献されたオープンソースプロジェクトです。プロジェクトに新機能を追加してくれたすべての貢献者と、貴重なフィードバックを提供してくれたユーザーに感謝したい。私たちは、このツールキットとベンチマークが、InternLM をファインチューニングし、独自のモデルを開発するための柔軟で効率的なコードツールをコミュニティに提供し、オープンソースコミュニティに継続的に貢献できることを願っています。2 つのオープンソースプロジェクト、[flash-attention](https://github.com/HazyResearch/flash-attention) と [ColossalAI](https://github.com/hpcaitech/ColossalAI) に感謝します。

## ライセンス

コードは Apache-2.0 でライセンスされており、モデルの重さは学術研究のために完全にオープンで、**無料** の商用利用も許可されています。商用ライセンスの申請は、[申請フォーム（英語）](https://wj.qq.com/s2/12727483/5dba/)/[申请表（中文）](https://wj.qq.com/s2/12725412/f7c1/)にご記入ください。その他のご質問やコラボレーションについては、<internlm@pjlab.org.cn> までご連絡ください。

## 引用

```
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM}},
    year={2023}
}
```
