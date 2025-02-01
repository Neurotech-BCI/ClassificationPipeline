import numpy as np
from mne_features.univariate import compute_samp_entropy, compute_spect_entropy
from scipy.signal import welch

### Wrapper for extracting temporal features from raw EEG samples in shape (num_channels,time_steps) and returning outputs in shape (num_channels, num_features) ###
class FeatureWrapper():
    def __init__(self):
        self.func_dict = {
            'spectral_entropy': self.compute_spectral_entropy,
            'sample_entropy': self.compute_sample_entropy,
            'alpha_bandpower': self.compute_alpha_bandpower,
            'hjorth_activity': self.compute_hjorth_activity,
            'hjorth_mobility': self.compute_hjorth_mobility,
            'hjorth_complexity': self.compute_hjorth_complexity
        }
    def compute_hjorth_activity(self,data,fs):
        activity = np.var(data, axis=1)
        return activity
    def compute_hjorth_mobility(self,data,fs):
        activity = self.compute_hjorth_activity(data,fs)
        mobility = np.sqrt(np.var(np.diff(data, axis=1), axis=1) / activity)
        return mobility
    def compute_hjorth_complexity(self,data,fs):
        mobility = self.compute_hjorth_mobility(data,fs)
        complexity = np.sqrt(np.var(np.diff(np.diff(data, axis=1), axis=1), axis=1) / np.var(np.diff(data, axis=1), axis=1)) / mobility
        return complexity
    def compute_alpha_bandpower(self, data, fs):  
        band=(8, 13)
        n_channels, _ = data.shape
        band_power = np.zeros(n_channels)
    
        for i in range(n_channels):
            freqs, psd = welch(data[i], fs=fs, nperseg=fs) 
            band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            band_power[i] = np.sum(psd[band_idx])
    
        return band_power

    def compute_spectral_entropy(self,data,sfreq):
        spectral_entropy = compute_spect_entropy(sfreq, data)
        return spectral_entropy
    def compute_sample_entropy(self,data,sfreq):
        sample_entropy = compute_samp_entropy(data)
        return sample_entropy
    
    def compute_features(self,data,sfreq,desired_features = ["alpha_bandpower"]):
        features = [self.func_dict[feature](data,sfreq) for feature in desired_features]
        features = np.stack(features,axis=1)
        return features
