#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>
# Please see the 'README.rst' and 'LICENSE' files in the SPORCO Extra
# repository for details of the copyright and user license

from scipy import signal

import numpy as np
import librosa
import os

SAMPLE_RATE = 16000
M = 4 # Number of elements (different durations) per note

min_midi = 60  # C4
max_midi = 83  # B5

# Resolution of dictionary element duration
t_res = 0.125

# Path to MAPS and chosen piano
HOME = os.path.expanduser('~')
MAPS_PATH = os.path.join(HOME, 'MAPS')
PIANO_DIR = 'ENSTDkCl'

# Save the dictionary under sporco/data
save_path = os.path.join('data', 'pianodict.npz')

# Create a reference to the directory with isolated notes
iso_dir = os.path.join(MAPS_PATH, PIANO_DIR, 'ISOL', 'NO')
# Obtain all of the files under this directory
files = os.listdir(iso_dir)
# Remove the extensions from the file names (.wav, .txt, .mid) and collapse duplicates
names = set([fname[:-4] for fname in files])

# Set span of midi notes to cover in the dictionary
span = max_midi - min_midi + 1

# Set the truncated lengths for each note
t_lengths = t_res * np.arange(1, M + 1)

# Initialize the dictionary
elems = {}

# Loop through all file names
for name in names:
    # Determine the midi number the name corresponds to
    num_midi = int(name[19:-9])
    # Check to make sure it is within the range
    if num_midi >= min_midi and num_midi <= max_midi:
        # Re-construct the path to the text file with (onset, offset, midi pitch)
        txt_path = os.path.join(iso_dir, name + '.txt')
        with open(txt_path) as note:
            note.readline()  # Throw away the first line (headers)
            # Read the respective values and convert to the correct data type
            onset, offset, midi_pitch = note.readline().strip('\n').split('\t')
            onset, offset, midi_pitch = float(
                onset), float(offset), int(midi_pitch)

        # Make sure the expected and retrieved midi note agree
        assert num_midi == midi_pitch

        # Re-construct the path to the audio file and read it in
        wav_path = os.path.join(iso_dir, name + '.wav')
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

        # Remove the audio before and after the occurrence of the note
        note_clip = audio[int(onset * SAMPLE_RATE): int(offset * SAMPLE_RATE)]

        # Create an entry in the dictionary for the midi note
        elems[num_midi] = []

        # Loop through each truncated length
        for t in t_lengths:
            # Determine the number of samples to grab
            smps = int(t * SAMPLE_RATE)
            # Determine the padding necessary for same length as maximum truncated length
            pad_amt = int(t_lengths[-1] * SAMPLE_RATE) - smps
            # Truncate the note to the specified length and perform RMS normalization
            elem_clip = note_clip[:smps] / \
                np.sqrt(np.sum(note_clip[:smps] ** 2) / smps)
            # Create envelope to ease hard truncation
            envelope = signal.tukey(elem_clip.size, 0.25)
            envelope[:envelope.size // 2] = 1
            elem_clip = elem_clip * envelope  # Modulate the element
            elem = np.append(elem_clip, np.zeros(pad_amt))  # Pad the note
            elems[num_midi] += [elem]  # Add note to the dictionary

# Convert the dictionary to an array compatible with sporco
arr = np.array([elems[s] for s in sorted(elems.keys())]
               ).reshape(span * len(t_lengths), -1)

# Save the dictionary under sporco/data
np.savez(save_path, elems=arr)
