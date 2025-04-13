import pandas as pd 
import numpy as np 
from scipy.signal import butter, filtfilt
from scipy.stats import zscore
import uvicorn
from fastapi import FastAPI
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from classify import FeatureWrapper
from io import StringIO
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
model: SVC = None
scaler: MinMaxScaler = None

class CSVInput(BaseModel):
    csv_data: str

@app.on_event("startup")
def load_model():
    global model, scaler
    model_path = os.getenv("MODEL_PATH", "binary_model.joblib")
    scaler_path = os.getenv("SCALER_PATH", "binary_scaler.joblib")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs 
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def preprocess(block, sfreq = 125):
    eeg_data = block.iloc[:, 1:17].to_numpy().T * 1e-6
    eeg_data = eeg_data[:,1250:11250]
    eeg_data = np.array(np.split(eeg_data,10,axis=-1))
    filtered = bandpass_filter(eeg_data, 0.5, 40, sfreq)
    return filtered

def predict(samples):
    global model, scaler
    selected_channels = [i for i in range(16)]
    desired_features = ["alpha_bandpower", "beta_bandpower", "theta_bandpower"]
    processed_samples = []
    wrapper = FeatureWrapper()
    for i, sample in enumerate(samples):
        features = wrapper.compute_features(sample,i,125,selected_channels,desired_features=desired_features)
        processed_samples.append(features)
    processed_samples = np.array(processed_samples)
    processed_samples = np.reshape(processed_samples,(processed_samples.shape[0],-1))
    processed_samples = scaler.transform(processed_samples)
    preds = model.predict(processed_samples)
    print(preds)
    prediction = np.argmax(np.bincount(preds))
    return prediction

@app.post("/inference")
def inference(data: CSVInput):
    try:
        df = pd.read_csv(StringIO(data.csv_data), header=None, sep = '\t')
        preprocessed_samples = preprocess(df)
        print(preprocessed_samples.shape)
        prediction = int(predict(preprocessed_samples))
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}