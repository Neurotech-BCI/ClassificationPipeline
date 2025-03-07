# Documentation 
## Repo Map
* classify.py: Main classifier functions with entrypoints for sklearn or pytorch classifier with automated cross-validation evaluation. Expects inputs and labels X, y in shape (num_samples, num_channels, num_features) and (num_samples,). Labels should be encoded with values 0 to n-1 where n is the number of unique classes. Also expects a constructed sklearn or pytorch classifier. Returns dictionary of evaluation results.
* feature.py: Feature extraction wrapper class for computing desired features. in FeatureWrapper.compute_features(), expects an EEG sample in the shape (num_channels, num_timesteps), the sampling frequency, and desired features specified with a list of keys to the feature function dictionary. Returns 2D numpy array for the given sample in shape (num_channels, num_features).
* hyperparameter.py: Hyperparameter optimization framework using Bayesian Optimization to find optimal channel and feature subset to maximize crossfold accuracy on EEG training dataset.
* eegnet.py: Implementation of a lightweight CNN for EEG classification. If not specified in constructor, kernel parameters are calculated depending on input length. This model works as input for the pytorch classifier. Works best on raw signal without feature extractions, so pass inputs to classification function as (num_samples, num_channels, num_timesteps).
* example_script.ipynb: Example notebook walking through steps for loading an example toy dataset with binary labels for relaxation or concentration, formatting it for the feature extraction and classifier, and getting cross fold evaluation results.
* extract_erps: Directory storing module for extracting ERPs from filepaths.

## Getting Started 
* Create new conda environment: 
```
conda create -n "env_name"
```
* Install requirements:
```
pip install -r requirements.txt
```
* Use example:
```
import numpy as np 
from sklearn.svm import SVC
from classify import classify_sklearn, FeatureWrapper

num_samples = 10
num_channels = 16
num_timesteps = 256
sfreq = 256

samples = np.random.rand(num_samples,num_channels,num_timesteps) # numpy array of random floats in shape (10,16,256)
labels = np.random.randint(0,2,size=(num_samples,)) # numpy array of randomly ordered 0s and 1s in shape (10,)

wrapper = FeatureWrapper()
processed_samples = np.array([wrapper.compute_features(sample, i, sfreq, [j for j in range(sample.shape[0])], desired_features=['alpha_bandpower']) for i, sample in enumerate(samples)])

metrics_dict = classify_sklearn(processed_samples,labels,SVC(),return_preds=False)
print(f"Mean CV accuracy: {metrics_dict['mean_accuracy']}")
print(f"Best CV accuracy: {metrics_dict['best_accuracy']}")
print(f"Worst CV accuracy: {metrics_dict['worst_accuracy']}")
print()
print(f"Mean CV precision: {metrics_dict['mean_precision']}")
print(f"Best CV precision: {metrics_dict['best_precision']}")
print(f"Worst CV precision: {metrics_dict['worst_precision']}")
print()
print(f"Mean CV recall: {metrics_dict['mean_recall']}")
print(f"Best CV recall: {metrics_dict['best_recall']}")
print(f"Worst CV recall: {metrics_dict['worst_recall']}")
print()
print(f"Mean CV F1: {metrics_dict['mean_f1']}")
print(f"Best CV F1: {metrics_dict['best_f1']}")
print(f"Worst CV F1: {metrics_dict['worst_f1']}")
```

## Extracting ERPs
* Use Example:
```
from extract_erps import process_data
eeg_file_paths = [r"data/Sample1.csv"]
metadata_file_path = [r"metadata/Sample1.csv"]
output_dir_name = "outputs"
process_data(eeg_file_paths,metadata_file_path, output_dir_name=output_dir_name, onset_time = 0, after_time = 1.0, sr = 125)
```



