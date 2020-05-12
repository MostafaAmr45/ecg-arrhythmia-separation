import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import librosa.display
import tkinter
import tkinter.filedialog
import os

#############################################
# Obtain ecg sample from csv file using pandas
root = tkinter.Tk()
root.withdraw()  # use to hide tkinter window

myAudioFile = tkinter.filedialog.askopenfilename()

dataset = pd.read_csv(myAudioFile)
y = [e for e in dataset.ECG]
y = np.array(y)
sr = 1000  # sampling frequency is taken from the source of the data set

# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

#######################################
# We'll compare frames using cosine similarity, and aggregate similar frames
# by taking their (per-frequency) median value.
S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
S_filter = np.minimum(S_full, S_filter)

##############################################
# The raw filter output can be used as a mask,
# but it sounds better if we use soft-masking.
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full

# multiply the magnitude component with phase to restore the wav files
# of the vocals and music

D_foreground = S_foreground * phase
D_background = S_background * phase

y_foreground = librosa.istft(D_foreground)
y_background = librosa.istft(D_background) * 1.5

##########################################
# Plot waveforms
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(y_background)
plt.title('ECG with arrhythmia')

plt.subplot(3, 1, 2)
plt.plot(y)
plt.title('Pure ECG')

plt.subplot(3, 1, 3)
plt.plot(y_foreground)
plt.title('Pure arrhythmia')
axes = plt.gca()
axes.set_ylim([-0.05, 0.05])
plt.tight_layout()
plt.show()
