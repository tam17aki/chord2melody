# chord2melody
## 概要
本リポジトリは「AIミュージックバトル！『弁財天』」から提供されたスターターキット ver 1.0 の改良版を提供する．改良版はディレクトリごとに分けている．

- cvae/ : 生成モデルを条件付き変分オートエンコーダ (Condtional Variational AutoEncoder; CVAE) へと拡張した．実装はRui Konuma氏の[実装](https://github.com/konumaru/benzaiten/tree/main)を参考にした． エンコーダ・デコーダはLSTMのままであるが，それらへの条件付けはコード（chord）と調（key）の情報により与えた．

## ライセンス
MIT licence.

- Copyright (C) 2023 Akira Tamamori
- Copyright (C) 2022 Rui Konuma
- Copyright (C) 2022 北原 鉄朗 (Tetsuro Kitahara)

## 依存パッケージ
実装はUbuntu 22.04上でテストした。Pythonのパージョンは`3.10.6`である。

- torch
- joblib
- midi2audio
- hydra-core
- progressbar2
- numpy
- scipy
- matplotlib

Ubuntuで動かす場合、**FluidSynthに関するサウンドフォントが必要**なので入れておく。

```bash

apt-get install fluidsynth

```

## 動かし方

|ファイル名|機能|
|---|---|
|preprocess.py | 前処理を実施するスクリプト|
|training.py |モデルの訓練を実施するスクリプト|
|generate.py | 訓練済のモデルを用いてメロディを合成するスクリプト|

各種の設定はyamlファイル（config.yaml）に記述する。

>
1. config.yamlの編集
2. preprocess.py による前処理の実施
3. training.pyによるモデル訓練の実施
4. generate.pyによるメロディ合成の実施

<u>preprocess.pyは一度だけ動かせばよい</u>。preprocess.pyにはモデル訓練に用いるMusicXML群（下記参照）のダウンロードや、それらからの特徴量抽出、またスターターキットが提供する伴奏データ・コード進行データのダウンロードが含まれる。

generate.pyには合成結果のMIDIファイルへの書き出し、Wavファイルへのポートが含まれる。

### 使用データ
訓練データは以下のサイトから入手可能なMusicXMLを用いる。

https://homepages.loria.fr/evincent/omnibook/

Ken Deguernel, Emmanuel Vincent, and Gerard Assayag.
"Using Multidimensional Sequences for Improvisation in the OMax Paradigm",
in Proceedings of the 13th Sound and Music Computing Conference, 2016.
