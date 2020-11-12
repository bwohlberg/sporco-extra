#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>
# Please see the 'README.rst' and 'LICENSE' files in the SPORCO Extra
# repository for details of the copyright and user license

"""
Single-channel CSC With Inhibition
==================================

This example demonstrates solving a convolutional sparse coding problem with a musical signal

  $$\mathrm{argmin}_\mathbf{x} \; \frac{1}{2} \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{m} - \mathbf{s} \right\|_2^2
  + \lambda \sum_m \| \mathbf{x}_{m} \|_1 + \sum_m \mathbf{\omega}^T_m \| \mathbf{x}_m \| + \sum_m \mathbf{z}^T_m \|
  \mathbf{x}_m \| \;,$$

where $\mathbf{d}_{m}$ is the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_{m}$ is the coefficient map corresponding
to the $m^{\text{th}}$ dictionary filter, $\mathbf{s}$ is the input image, and $\mathbf{\omega}^T_m$ and $\mathbf{z}^T_m$
are inhibition weights corresponding to lateral and self inhibition, respectively. (See cbpdnin.ConvBPDNInhib)
"""


from __future__ import print_function
import matplotlib.pyplot as plt
from builtins import input
from builtins import range

import numpy as np
import librosa
import os

from sporco.cupy import np2cp, cp2np
from sporco import plot
import sporco.metric as sm


try:
    import cupy as cp
    try:
        from sporco.cupy.admm import cbpdnin
        cp.cuda.Device(0).compute_capability
        CUPY = True
    except cp.cuda.runtime.CUDARuntimeError:
        from sporco.admm import cbpdnin
        CUPY = False
        print("GPU device inaccessible, CPU will be used instead")
except ImportError:
    from sporco.admm import cbpdnin
    CUPY = False
    print("cupy not installed, CPU will be used")


"""
Load example piano piece.
"""

audio, fs = librosa.load(os.path.join('data',
    'MAPS_MUS-deb_clai_ENSTDkCl_excerpt.wav'), 16000)


"""
Load dictionary and display it. Grouped elements are displayed on the same axis.
"""

M = 4
D = np.load(os.path.join('data', 'pianodict.npz'))['elems']
fig = plot.figure(figsize=(10, 16))
for i in range(24):
    plot.subplot(8, 3, i + 1, title=f'{librosa.midi_to_note(i+60)}',
                 ylim=[-6, 6])
    plot.gca().get_xaxis().set_visible(False)
    plot.gca().get_yaxis().set_visible(False)
    plot.gca().set_frame_on(False)
    for k in range(M):
        plot.plot(D[M*(i+1)-k-1], fig=fig)
fig.show()


"""
Set :class:`.admm.cbpdn.ConvBPDNInhib` solver options.
"""

lmbda = 1e-1
mu = 1e1
gamma = 1e0
opt = cbpdnin.ConvBPDNInhib.Options({'Verbose': True, 'MaxMainIter': 500,
                                     'RelStopTol': 5e-2, 'AuxVarObj': False,
                                     'rho': 100, 'AutoRho': {'Enabled': False}})


"""
Initialise and run CSC solver.
"""

Wg = np.eye(24)
Wg = np.repeat(Wg, M, axis=1)

if CUPY:
    D = np2cp(D)
    Wg = np2cp(Wg)
    audio = np2cp(audio)

b = cbpdnin.ConvBPDNInhib(D.T, audio, Wg, int(
    0.20*16000), None, lmbda, mu, gamma, opt, dimK=None, dimN=1)
X = b.solve()

if CUPY:
    X = cp2np(X)

print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'))


"""
Reconstruct image from sparse representation.
"""

recon = b.reconstruct().squeeze()

if CUPY:
    audio = cp2np(audio)
    recon = cp2np(recon)

print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(audio, recon))


"""
Show activation map across time. You may have to zoom in to see well-localized activations.
"""

wl, hl = 160, 80

# Get activation magnitude and apply padding in preparation for frame conversion
X = X.squeeze().T
# Keep track of the original length
len_orig = X.shape[-1]
# Number of frames available
num_frms = (len_orig - 1) // hl + 1
# Pad the number of activations to end on a full frame
padding = hl * (num_frms - 1) + wl - len_orig
frm_X = np.append(np.abs(X), np.zeros((X.shape[0], padding)), axis=1)

# Frame-ify the activations
N = frm_X.shape[-1]
frm_X = np.reshape(frm_X, (-1, M, N))
# Re-structure the activations so the samples of each frame are isolated
# We now have note x element x frame x sample
frm_X = np.concatenate([np.expand_dims(frm_X[:, :, n * hl: n * hl + wl], axis=2)
                        for n in range(num_frms)], axis=2)
# Sum the activation samples across each frame
frm_X = np.sum(frm_X, axis=-1)

frm_X = np.reshape(frm_X, (24 * M, -1))
frm_X = frm_X / (np.max(frm_X) + 1e-8)
frm_X = librosa.amplitude_to_db(frm_X)
times = np.arange(num_frms) * hl / 16000

# Plot the activation map
fig = plt.figure(figsize=(15, 10))
x, y = np.meshgrid(times, np.arange(frm_X.shape[0]))
plt.pcolormesh(x, y, frm_X)
plt.colorbar(format='%+2.0f dB')

plt.title('Activation Map')
plt.xlabel('Time (s)')
plt.ylabel('Elements')
ytick_labs = librosa.midi_to_note(np.arange(60, 84))
ytick_locs = (np.arange(len(ytick_labs))) * M
plt.yticks(ytick_locs, ytick_labs)
plt.tick_params(axis='y', which='major', labelsize=6)
plt.grid(True, color='black', linestyle='--')
fig.show()


"""
Show the ground-truth note activity
"""

fig = plot.figure(figsize=(14, 7))
txt_path = os.path.join('data', 'MAPS_MUS-deb_clai_ENSTDkCl_excerpt.txt')
with open(txt_path) as notes:
    notes.readline()  # Throw away the first line (headers)
    for note in notes:
        # Read the respective values and convert to the correct data type
        onset, offset, midi_pitch = note.strip('\n').split('\t')
        onset, offset, midi_pitch = float(
            onset), float(offset), int(midi_pitch)
        plot.plot([midi_pitch] * 2, [onset, offset],
                  linewidth=10, color='black', fig=fig)
ytick_labs = librosa.midi_to_note(np.arange(60, 84))
ytick_locs = np.arange(60, 84)
plt.ylim([59, 84])
plt.yticks(ytick_locs, ytick_labs)
plt.grid(True, color='black', linestyle='--')
plt.title('Ground Truth Pianoroll')
fig.show()


"""
Display original and reconstructed signal. Most offsets are missing since elements are truncated.
This problem can be alleviated by increasing the extent of each note group during dictionary creation.
"""

fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.plot(audio, title='Original', fig=fig)
plot.subplot(1, 2, 2)
plot.plot(recon, title='Reconstructed', fig=fig)
fig.show()


"""
Get iterations statistics from solver object and plot functional value, ADMM primary and
dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
"""

its = b.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'], fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
fig.show()


# Wait for enter on keyboard
input()
