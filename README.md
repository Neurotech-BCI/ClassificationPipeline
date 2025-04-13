# Documentation 
## Repo Map
* inference_api.py: Model inference API that loads the trained model specified in .env on startup and responds to post requests passing string-formatted csvs of EEG data and returns the model's prediction.
* test_inference.py: Test script that sends a request to the endpoint in inference_api using an example data csv.
* train.py: Script to train the model on 2-class or 3-class classification and save to /models.
* classify.py: Main classifier functions with entrypoints for sklearn or pytorch classifier with automated cross-validation evaluation. Expects inputs and labels X, y in shape (num_samples, num_channels, num_features) and (num_samples,). Labels should be encoded with values 0 to n-1 where n is the number of unique classes. Also expects a constructed sklearn or pytorch classifier. Returns dictionary of evaluation results.
* feature.py: Feature extraction wrapper class for computing desired features. in FeatureWrapper.compute_features(), expects an EEG sample in the shape (num_channels, num_timesteps), the sampling frequency, and desired features specified with a list of keys to the feature function dictionary. Returns 2D numpy array for the given sample in shape (num_channels, num_features).
* eegnet.py: Implementation of a bayesian CNN for EEG classification. If not specified in constructor, kernel parameters are calculated depending on input length. This model works as input for the pytorch classifier. Works best on raw signal without feature extractions, so pass inputs to classification function as (num_samples, num_channels, num_timesteps).
* example_script.ipynb: Example notebook walking through data loading and evaluation of ML classifiers in different conditions.
* preprocess.ipynb: Preprocessing workflow to prepare training data from raw csv uploads.

## Getting Started 
* Create new conda environment: 
```
conda create -n "env_name"
```
* Install requirements:
```
pip install -r requirements.txt
```
* Start server at localhost http://127.0.1:8000"
```
uvicorn inference_api:app --reload
```
* Test inference at the /inference endpoint:
```
python test_inference.py
```
## Using the Classify Module:
```
import numpy as np 
from sklearn.ensemble import AdaBoostClassifier
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




