# -*- coding: utf-8 -*-
"""Demonstration script for melody generate using pretrained model.

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
import csv
import os

import midi2audio
import mido
import music21
import numpy as np
import torch
from hydra import compose, initialize

from model import MelodyComposer


class PianoRoll:
    """Class for Piano-Roll."""

    def __init__(self, cfg):
        """Initialize class.

        cfg (Config) : configures for encoder.
        """
        self.cfg = cfg
        self.piano_roll = None
        self.note_nums = None
        self.duration = None

    def _make_chord_prog(self):
        """Return chord progress (ChordSymbol objects) read from chord csv file."""
        melody_length = self.cfg.feature.melody_length
        n_beats = self.cfg.feature.n_beats
        chord_prog = [None] * int(melody_length * n_beats)  # 8 * 4 = 32
        chord_file = os.path.join(
            self.cfg.benzaiten.root_dir,
            self.cfg.benzaiten.adlib_dir,
            self.cfg.demo.chord_file,
        )
        with open(chord_file, encoding="utf-8") as file_handler:
            reader = csv.reader(file_handler)
            for row in reader:
                measure_id = int(row[0])
                if measure_id < melody_length:
                    beat_id = int(row[1])
                    chord_prog[measure_id * 4 + beat_id] = music21.harmony.ChordSymbol(
                        root=row[2], kind=row[3], bass=row[4]
                    )
        for i, _chord in enumerate(chord_prog):
            if _chord is not None:
                chord = _chord
            else:
                chord_prog[i] = chord
        return chord_prog

    def _chord_to_chroma(self, chord_prog):
        """Convert seq. of ChordSymbols into seq. of many-hot (chroma) vectors."""
        division = self.cfg.feature.n_beats  # how many chords exit in a single measure
        n_beats = self.cfg.feature.n_beats
        beat_reso = self.cfg.feature.beat_reso
        time_length = int(n_beats * beat_reso / division)  # 4 * 4 / 4 = 4
        seq_chord = [None] * (time_length * len(chord_prog))  # 4 * 32 = 128
        for i, chord in enumerate(chord_prog):
            for _t in range(time_length):
                if isinstance(chord, music21.harmony.ChordSymbol):
                    seq_chord[i * time_length + _t] = chord
                else:
                    seq_chord[i * time_length + _t] = music21.harmony.ChordSymbol(chord)

        chroma = np.zeros((len(seq_chord), 12))  # [128, 128]
        for i, chord in enumerate(seq_chord):
            if chord is not None:
                for note in chord.notes:
                    chroma[i, note.pitch.midi % 12] = 1
        return chroma

    @torch.no_grad()
    def generate(self, model, device):
        """Generate a piano-roll from pretrained model and chord symbols.

        Adopted from 'generate_pianoroll' by Rui Konuma.

        NOTE: piano-roll is a numpy ndarray object.
        """
        var = {"progress": None, "chord_chroma": None, "mode": None}
        var["progress"] = self._make_chord_prog()
        var["chord_chroma"] = self._chord_to_chroma(var["progress"])
        var["mode"] = np.zeros((var["chord_chroma"].shape[0], 1))  # major
        if self.cfg.feature.key_mode == "minor":
            var["mode"] = np.ones((var["chord_chroma"].shape[0], 1))  # minor
        condition = np.concatenate((var["chord_chroma"], var["mode"]), axis=1)
        inputs = torch.from_numpy(condition.astype(np.float32)).unsqueeze(0).to(device)
        melody_length = inputs.shape[1]  # 128
        batch_size = self.cfg.feature.unit_measures
        min_note_num = self.cfg.feature.notenum_from
        max_note_num = self.cfg.feature.notenum_thru
        piano_roll = np.zeros((melody_length, max_note_num - min_note_num + 1))
        for i in range(0, melody_length, batch_size):
            latent_rand = torch.randn(1, self.cfg.model.decoder.latent_dim).to(device)
            y_new = model.decode(latent_rand, inputs[:, i : i + batch_size])
            y_new = y_new.softmax(dim=2).cpu().detach().numpy()
            piano_roll[i : i + batch_size, :] = y_new[0]

        self.piano_roll = piano_roll

    def _convert_notenum(self):
        """Convert piano-roll into sequence of note numbers."""
        note_nums = []
        notenum_from = self.cfg.feature.notenum_from
        for i in range(self.piano_roll.shape[0]):
            num = np.argmax(self.piano_roll[i, :])
            note_num = -1 if num == 0 else num + notenum_from - 1
            note_nums.append(note_num)

        note_length = len(note_nums)
        duration = [1] * note_length
        for i in range(note_length):
            k = 1
            while i + k < note_length:
                if note_nums[i] > 0 and note_nums[i] == note_nums[i + k]:
                    note_nums[i + k] = 0
                    duration[i] += 1
                else:
                    break
                k += 1
        self.note_nums = note_nums
        self.duration = duration

    def export_midi(self):
        """Make Midi object from piano-roll."""
        beat_reso = self.cfg.feature.beat_reso
        n_beats = self.cfg.feature.n_beats
        transpose = self.cfg.feature.transpose
        intro_blank_measures = self.cfg.feature.intro_blank_measures

        backing_file = os.path.join(
            self.cfg.benzaiten.root_dir,
            self.cfg.benzaiten.adlib_dir,
            self.cfg.demo.backing_file,
        )
        midi = mido.MidiFile(backing_file)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # convert piano-roll into sequence of note numbers.
        self._convert_notenum()

        var = {
            "init_tick": intro_blank_measures * n_beats * midi.ticks_per_beat,
            "cur_tick": 0,
            "prev_tick": 0,
        }
        for i, notenum in enumerate(self.note_nums):
            if notenum > 0:
                var["cur_tick"] = (
                    int(i * midi.ticks_per_beat / beat_reso) + var["init_tick"]
                )
                track.append(
                    mido.Message(
                        "note_on",
                        note=notenum + transpose,
                        velocity=100,
                        time=var["cur_tick"] - var["prev_tick"],
                    )
                )
                var["prev_tick"] = var["cur_tick"]
                var["cur_tick"] = (
                    int((i + self.duration[i]) * midi.ticks_per_beat / beat_reso)
                    + var["init_tick"]
                )
                track.append(
                    mido.Message(
                        "note_off",
                        note=notenum + transpose,
                        velocity=100,
                        time=var["cur_tick"] - var["prev_tick"],
                    )
                )
                var["prev_tick"] = var["cur_tick"]

        return midi


def main(cfg):
    """Perform ad-lib melody generation."""
    # setup network and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chkpt_dir = os.path.join(cfg.benzaiten.root_dir, cfg.demo.chkpt_dir)
    checkpoint = os.path.join(chkpt_dir, cfg.demo.chkpt_file)
    model = MelodyComposer(cfg.model, device).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()  # turn on eval mode

    # generate ad-lib melody and export the result to midi format
    piano_roll = PianoRoll(cfg)
    piano_roll.generate(model, device)
    midi = piano_roll.export_midi()
    midi_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.midi_file
    )
    midi.save(midi_file)

    # export midi to wav
    wav_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.wav_file
    )
    fluid_synth = midi2audio.FluidSynth(sound_font=cfg.demo.sound_font)
    fluid_synth.midi_to_audio(midi_file, wav_file)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
