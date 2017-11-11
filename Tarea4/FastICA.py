"""
=====================================
Blind source separation using FastICA
=====================================

An example of estimating sources from noisy data.

:ref:`ICA` is used to estimate sources given noisy measurements.
Imagine 3 instruments playing simultaneously and 3 microphones
recording the mixed signals. ICA is used to recover the sources
ie. what is played by each instrument. Importantly, PCA fails
at recovering our `instruments` since the related signals reflect
non-Gaussian processes.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scikits.audiolab import wavread,wavwrite
from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data
np.random.seed(0)

rec1, fs1, enc1 = wavread('rsm2_mA.wav')
rec2, fs2, enc2 = wavread('rsm2_mB.wav')

S = np.c_[rec1, rec2]

# Standardize data
S /= S.std(axis=0)

# Compute ICA
ica = FastICA(n_components=2)
S_ = ica.fit_transform(S)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(S, np.dot(S_, A_.T) + ica.mean_)

max_source, min_source = 1.0, -1.0
max_result, min_result = max(S_.flatten()), min(S_.flatten())
S_ = map( lambda x: (2.0 * (x - min_result))/(max_result - min_result) + -1.0, S_.flatten() )
S_ = np.reshape( S_, (np.shape(S_)[0] / 2, 2) )
	
# Store the separated audio, listen to them later
wavwrite(S_[:,0], 'separated_1.wav', fs1, enc1)
wavwrite(S_[:,1], 'separated_2.wav', fs2, enc2)



