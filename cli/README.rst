Convolutional Lateral Inhibition
================================
:Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

This directory contains the resources necessary to test convolutional sparse coding with lateral inhibition on a characteristic example where an audio signal representing a small piano excerpt is encoded using the activations of individual keys. The encoding provides the information necessary to perform automatic music transcription (AMT), whereby the pitch, onset, and offset of each note in a music signal is estimated. Here, we encode an audio excerpt from the MAPS_ dataset, and draw a comparison to its ground-truth note activity. The MAPS_ dataset is only required if ones wishes to regenerate the dictionary or test out different audio excerpts. We have provided the necessary resources to complete the simple example outlined here.

.. _MAPS: https://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/


`maps_dict_gen.py <maps_dict_gen.py>`_
  Script that generates the dictionary `data/pianodict.npz <data/pianodict.npz>`_ from scratch. The ENSTDkCL partition of the MAPS_ dataset is required to regenerate `data/pianodict.npz <data/pianodict.npz>`_ from scratch using this script. Here, the dictionary elements are obtained from the recordings of isolated piano notes, which are truncated to different lengths, in order to represent different possible durations of the same note. This is exactly why we need lateral inhibition - because the same pitch can only ever be active once on the piano, but we have multiple elements representing the same pitch.

  If the MAPS_ dataset is downloaded and properly referenced at the top of the script, the script can simply be run. Some dictionary generation parameters, such as the number of elements per pitch, the resolution of time duration of the elements, and the MIDI range of the dictionary, can also be modified at the top of the script.


`cbpdnin_msc.py <cbpdnin_msc.py>`_
  Example script that encodes the audio with convolutional sparse coding with lateral inhibition using the dictionary `data/pianodict.npz <data/pianodict.npz>`_. The elements are grouped by pitch, such that a solution with multiple concurrent activations of the same pitch is highly discouraged. The script also displays the activations of each element over time (a proxy for our transcription), along with the ground truth piano-key activations. Note that this script requires installation of the `librosa` package in addition to `sporco` and its dependencies.


`data/MAPS_MUS-deb_clai_ENSTDkCL_excerpt.wav <data/MAPS_MUS-deb_clai_ENSTDkCL_excerpt.wav>`_
  Audio we encode to exemplify convolutional sparse coding with lateral inhibition.


`data/MAPS_MUS-deb_clai_ENSTDkCL_excerpt.txt <data/MAPS_MUS-deb_clai_ENSTDkCL_excerpt.txt>`_
  Ground-truth containing the pitch, onset, and offset of each piano note present in the audio excerpt.


`data/pianodict.npz <data/pianodict.npz>`_
  Dictionary generated using the default parameters (hard-coded in `maps_dict_gen.py <maps_dict_gen.py>`_), provided for convenience.
