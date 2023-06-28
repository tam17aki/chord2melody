# -*- coding: utf-8 -*-
"""Preprocess script.

Copyright (C) 2022 by 北原 鉄朗 (Tetsuro Kitahara)
Copyright (C) 2022 Rui Konuma
Copyright (C) 2023 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import glob
import os
import subprocess

import joblib
import music21
import numpy as np
from hydra import compose, initialize
from progressbar import progressbar as prg


def make_sequence(data, max_seq_len, drop_last=True, pad_value=0):
    """Split an array into a sequence of sub-arrays."""
    _sequence = []
    for i in range(0, len(data), max_seq_len):
        row = list(data[i : (i + max_seq_len)])

        if len(row) == max_seq_len:
            _sequence.append(row)
        elif not drop_last:
            num_pad = max_seq_len - len(row)
            row = row + [pad_value] * num_pad
            _sequence.append(row)

    sequence = np.array(_sequence)
    return sequence


class MusicXMLFeature:
    """Class for MusicXML.

    This class is adopted from the repository of konumaru.
    """

    def __init__(self, cfg, xml_file):
        """Initialize class."""
        assert cfg.feature.key_root in ["C", "D", "E", "F", "G", "A", "B"]
        self.score = self._get_score(xml_file, cfg.feature.key_root)
        self.cfg = cfg
        self.notes, self.chords = self.get_notes_and_chords()

    def _get_score(self, xml_file: str, root: str):
        """Get an object of Score class (music21)."""
        # MusicXMLをパースしてScoreクラスのオブジェクトを取得する
        score = music21.converter.parse(xml_file, format="musicxml")

        # XMLを分析してキー（調）の情報を取得
        key = score.analyze("key")

        # 音程（インターバル）を計算 -> 2音の情報を与える必要あり
        # tonic: 主音; キー（調）の基礎となるスケール（音階）の出発点にあたる音
        #        -> C, B-, F, A-, E-, など分析結果に基づいてバラバラの値
        # root: "C" や "D"など キーのルート音; コードの一番低い音
        # 調の主音をオリジナルのtonicから指定したrootへと移したい（移調！）
        interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch(root))

        # 調の情報とインターバルに基づいて移調 (C major や A minorなど)
        # 学習に用いるデータの「調」を揃えることで，学習しやすくなる
        score.transpose(interval, inPlace=True)
        return score

    def get_mode(self):
        """Get mode（旋律）."""
        key = self.score.analyze("key")
        mode = "None" if key is None else str(key.mode)
        return mode

    def get_notes_and_chords(self):
        """Return a list of note numbers and a list of chords from score info."""
        notes = []  # type: ignore
        chords = []  # type: ignore

        num_beats = self.cfg.feature.n_beats  # 1小節を何拍で区切るか -> 4
        num_parts_of_beat = self.cfg.feature.beat_reso  # 1拍を何等分するか -> 4
        # -> 小節を4 * 4 = 16等分する

        # self.score: Score型のオブジェクト
        # self.score.parts: 曲のパート（各楽器）
        # self.score.parts[0]: ピアノ（メロディ担当）
        # ピアノパートのうち，"Measure"で指定されるものは「各小節」の情報
        # →getElementsByClass("Measure")により，小節の系列（イテレータ）を取得
        for measure in self.score.parts[0].getElementsByClass("Measure"):

            # 鍵盤情報を格納するリスト -> Noneで初期化
            m_notes = [None] * num_beats * num_parts_of_beat
            for note in measure.getElementsByClass("Note"):  # 1小節内の音符たち
                # noteは鍵盤位置に対応するもの (music21.note.Note) -> CとかGとか
                onset = note.offset  # 小節内の鳴り始め位置（浮動小数点）
                offset = onset + note.duration.quarterLength  # 鳴り終わり位置（浮動小数点）
                start_idx = int(onset * num_parts_of_beat)  # 4倍して時間解像度を調整
                end_idx = int(offset * num_parts_of_beat) + 1  # 4倍して時間解像度を調整
                end_idx = end_idx if end_idx < 16 else 16  # 小節内に収まるように調整
                num_item = int(end_idx - start_idx)  # 音符長を計算
                # 小節内の当該音符が鳴っているところをnote番号で塗りつぶす
                m_notes[start_idx:end_idx] = [note] * num_item
            notes.extend(m_notes)  # 当該小節のノート番号群を追加

            m_chords = [None] * num_beats * num_parts_of_beat
            for chord in measure.getElementsByClass("ChordSymbol"):  # 当該小節の各コード記号
                offset = chord.offset  # コードの絶対位置（浮動小数点）
                start_idx = int(offset * num_parts_of_beat)
                end_idx = int(num_beats * num_parts_of_beat) + 1  # 小節終端
                end_idx = end_idx if end_idx < 16 else 16
                num_item = int(end_idx - start_idx)
                # 小節終端まですべて単一のコードで埋める
                # →小節内に複数のコードが現れる場合はoffsetでずらした位置から上書き
                # →コードには「継続長」の情報がないため，この方法がベターというわけ
                m_chords[start_idx:end_idx] = [chord] * num_item
            chords.extend(m_chords)  # 当該小節のコード番号情報を追加

        return notes, chords

    def get_seq_notenum(self):
        """Return a sequence of note numbers."""
        min_note_num = self.cfg.feature.notenum_from
        seq_notenum = [
            # 鍵盤位置を具体的なMIDIのノート番号に変換する
            # → min_note_numからの相対的な値 (1 から 48)
            # 0 となるのは休符要素に対応させる（Noneはemptyに対応する）
            int(n.pitch.midi) - min_note_num + 1 if n is not None else 0
            for n in self.notes
        ]
        return np.array(seq_notenum)

    def get_seq_chord_chorma(self):
        """Return a sequence of many-hot vectors corresponding to note numbers."""
        onehot_chord_seq = np.zeros((len(self.chords), 12))
        for i, chord in enumerate(self.chords):
            if chord is not None:  # 小節内の当該位置におけるコード記号
                for note in chord.notes:  # ピッチクラスに分解（ドミソならC, E, G）
                    onehot_chord_seq[i, note.pitch.midi % 12] = 1
        return onehot_chord_seq


def get_music_xml(cfg):
    """Download Omnibook MusicXML data for training."""
    xml_url = cfg.preprocess.xml_url
    xml_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.xml_dir)
    os.makedirs(xml_dir, exist_ok=True)

    subprocess.run(
        "echo -n Download Omnibook MusicXML ... ", text=True, shell=True, check=False
    )

    command = "wget " + "-P " + "/tmp/" + " " + xml_url
    subprocess.run(command, text=True, shell=True, capture_output=True, check=False)

    zip_file = os.path.basename(xml_url)
    command = "cd " + xml_dir + "; " + "unzip " + "/tmp/" + zip_file
    subprocess.run(command, text=True, shell=True, capture_output=True, check=False)

    command = "mv " + xml_dir + "Omnibook\\ xml/*.xml " + xml_dir
    subprocess.run(command, text=True, shell=True, check=False)

    command = "rm -rf " + xml_dir + "Omnibook\\ xml"
    subprocess.run(command, text=True, shell=True, check=False)

    command = "rm -rf " + xml_dir + "__MACOSX"
    subprocess.run(command, text=True, shell=True, check=False)

    print(" done.")


def extract_features(cfg):
    """Extract features.

    This function is adopted from the repository of konumaru.
    """
    feats = {"notenum_all": [], "chord_chroma_all": [], "mode_all": []}

    xml_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.xml_dir)
    os.makedirs(xml_dir, exist_ok=True)
    for xml_file in prg(
        glob.glob(xml_dir + "/*.xml"), prefix="Extract features from MusicXML: "
    ):
        feat = MusicXMLFeature(cfg, xml_file)  # 鍵盤記号列およびコード記号列を扱うクラス

        mode_map = {"major": 0.0, "minor": 1.0}
        mode = feat.get_mode()  # モード（旋法）を取得 ("major" or "minor")

        seq_notenum = feat.get_seq_notenum()  # 鍵盤記号系列をノート番号列に変換
        seq_chord_chroma = feat.get_seq_chord_chorma()  # コード記号列をmany-hot列に変換

        # ノート番号列，one-hot列 (melody)，many-hot列 (chord)を切り分ける
        # 切り分けるときの部分系列の最大サイズはmax_seq_len
        max_seq_len = (
            # 4 * 4 * 4 = 64
            cfg.feature.unit_measures
            * cfg.feature.beat_reso
            * cfg.feature.n_beats
        )
        feat_notenum = make_sequence(seq_notenum, max_seq_len)
        feat_chord_chroma = make_sequence(seq_chord_chroma, max_seq_len)

        feats["notenum_all"].append(feat_notenum)
        feats["chord_chroma_all"].append(feat_chord_chroma)

        # 一様な"major" に対応する数値列 (0.0, 0.0, ..., 0.0) もしくは
        # 一様な"minor" に対応する数値列 (1.0, 1.0, ..., 1.0) の作成
        # [0.0] or [1.0]をnote番号長の個数だけ一列に敷き詰める
        feats["mode_all"].append(np.tile([mode_map[mode]], len(feat_notenum)))

    return (
        np.vstack(feats["notenum_all"]),
        np.vstack(feats["chord_chroma_all"]),
        np.concatenate(feats["mode_all"]),
    )


def save_features(cfg, notenum_all, chord_all, mode_all):
    """Save feature vectors."""
    feats_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.feat_dir)
    os.makedirs(feats_dir, exist_ok=True)
    feat_file = os.path.join(feats_dir, cfg.preprocess.feat_file)
    joblib.dump(
        {"notenum": notenum_all, "chord": chord_all, "mode": mode_all}, feat_file
    )
    print("Save extracted features to " + feat_file)


def get_backing_chord(cfg):
    """Download backing file (midi) and chord file (csv)."""
    g_drive_url = '"https://drive.google.com/uc?export=download&id="'
    adlib_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir)
    os.makedirs(adlib_dir, exist_ok=True)

    backing_url = g_drive_url + cfg.demo.backing_fid
    backing_file = os.path.join(adlib_dir, cfg.demo.backing_file)
    chord_url = g_drive_url + cfg.demo.chord_fid
    chord_file = os.path.join(adlib_dir, cfg.demo.chord_file)

    subprocess.run(
        "echo -n Download backing file for demo ... ",
        text=True,
        shell=True,
        check=False,
    )
    command = "wget " + backing_url + " -O " + backing_file
    subprocess.run(command, text=True, shell=True, capture_output=True, check=False)
    print(" done.")

    subprocess.run(
        "echo -n Download chord file for demo ... ", text=True, shell=True, check=False
    )
    command = "wget " + chord_url + " -O " + chord_file
    subprocess.run(command, text=True, shell=True, capture_output=True, check=False)
    print(" done.")


def main(cfg):
    """Perform preprocess."""
    # Download Omnibook MusicXML
    get_music_xml(cfg)

    # Extract features from MusicXML
    notenum_all, chord_all, mode_all = extract_features(cfg)

    # Save extracted features
    save_features(cfg, notenum_all, chord_all, mode_all)

    # Download backing file (midi) and chord file (csv)
    get_backing_chord(cfg)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
