import numpy as np
from mne_features.univariate import compute_samp_entropy, compute_spect_entropy, compute_hjorth_complexity, compute_hjorth_mobility, compute_kurtosis, compute_skewness
from scipy.signal import welch
from collections import defaultdict
from .graph_features import calculate_node_strength, local_efficiency, clustering_coefficient, betweenness_centrality

### Wrapper for extracting temporal features from raw EEG samples in shape (num_channels,time_steps) and returning outputs in shape (num_channels, num_features) ###
class FeatureWrapper():
    def __init__(self):
        self.func_dict = {
            #'spectral_entropy': self.compute_spectral_entropy,
            'sample_entropy': self.compute_sample_entropy,
            'alpha_bandpower': self.compute_alpha_bandpower,
            'beta_bandpower': self.compute_beta_bandpower,
            'theta_bandpower': self.compute_theta_bandpower,
            'hjorth_activity': self.compute_hjorth_activity,
            'hjorth_mobility': self.compute_hjorth_mobility,
            'hjorth_complexity': self.compute_hjorth_complexity,
            #'node_strength': self.compute_node_strength,
            #'local_efficiency': self.compute_local_efficiency,
            #'clustering_coefficient': self.compute_clustering_coefficient,
            #'betweenness_centrality': self.compute_betweenness_centrality
            #'kurtosis': self.compute_kurtosis,
            #'skewness': self.compute_skewness
        }
        self.cache = []
    def compute_skewness(self,data,fs):
        skewness = compute_skewness(data)
        return skewness
    def compute_kurtosis(self,data,fs):
        kurtosis = compute_kurtosis(data)
        return kurtosis
    def compute_hjorth_activity(self,data,fs):
        activity = np.var(data, axis=1)
        return activity
    def compute_hjorth_mobility(self,data,fs):
        #activity = self.compute_hjorth_activity(data,fs)
        #mobility = np.sqrt(np.var(np.diff(data, axis=1), axis=1) / activity)
        mobility = compute_hjorth_mobility(data)
        return mobility
    def compute_hjorth_complexity(self,data,fs):
        #mobility = self.compute_hjorth_mobility(data,fs)
        #complexity = np.sqrt(np.var(np.diff(np.diff(data, axis=1), axis=1), axis=1) / np.var(np.diff(data, axis=1), axis=1)) / mobility
        complexity = compute_hjorth_complexity(data)
        return complexity
    def compute_alpha_bandpower(self, data, fs):  
        band=(8, 13)
        n_channels = data.shape[0]
        band_power = np.zeros(n_channels)
    
        for i in range(n_channels):
            freqs, psd = welch(data[i], fs=fs, nperseg=fs) 
            band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            band_power[i] = np.sum(psd[band_idx])
    
        return band_power
    def compute_beta_bandpower(self, data, fs):  
        band=(12, 30)
        n_channels = data.shape[0]
        band_power = np.zeros(n_channels)
    
        for i in range(n_channels):
            freqs, psd = welch(data[i], fs=fs, nperseg=fs) 
            band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            band_power[i] = np.sum(psd[band_idx])
    
        return band_power
    def compute_theta_bandpower(self, data, fs):  
        band=(3.5, 7.5)
        n_channels = data.shape[0]
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
    
    def compute_node_strength(self, data, fs): #returns shape(num_imfs, num channels)
        ns = calculate_node_strength(data)
        return ns
    
    def compute_local_efficiency(self, data, fs): #returns shape(num_imfs, num channels)
        le = local_efficiency(data)
        return le
    
    def compute_clustering_coefficient(self, data, fs): #returns shape(num_imfs, num channels)
        cc = clustering_coefficient(data)
        return cc
    
    def compute_betweenness_centrality(self, data, fs): #returns shape(num_imfs, num channels)
        bc = betweenness_centrality(data)
        return bc





    
    def compute_features(self,data,data_idx,sfreq,channel_indices,desired_features = ["alpha_bandpower"]):
        if len(self.cache) <= data_idx:
            self.cache.append(defaultdict(dict))
        features = [[] for _ in range(len(channel_indices))]
        for i, channel in enumerate(channel_indices):
            for feature in desired_features:
                #print((np.expand_dims(data[channel],axis=0)).shape)
                if feature in self.cache[data_idx][channel]:
                    #print(f"{feature} is cached for channel {channel}.")
                    features[i].append(self.cache[data_idx][channel][feature])
                else:
                    #print(f"Calculating {feature} for channel {channel}.")
                    datum = np.expand_dims(data[channel],axis=0)
                    calculated_feature = (self.func_dict[feature](datum,sfreq)).item()
                    #print(calculated_feature)
                    features[i].append(calculated_feature)
                    self.cache[data_idx][channel][feature] = calculated_feature
        features = np.array(features)
        #print(f"Features shape: {features.shape}")
        return features
