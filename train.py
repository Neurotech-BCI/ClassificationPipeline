import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from classify import FeatureWrapper
from sklearn.preprocessing import MinMaxScaler
import joblib 

import numpy as np
def get_dataset(names,samples,classes=2,cross=False):
    data = []
    labels = []
    for name, sample in zip(names,samples):
        for s in sample:
            datum = np.load(f"data/preprocessed/data/{name}_{s}.npy")
            if classes == 2:
                datum = np.concatenate((datum[:10],datum[40:]))
                label = np.concatenate(([0 for _ in range(10)],[1 for _ in range(10)]))
            elif classes == 3:
                datum = datum = np.concatenate((datum[:10], datum[10:20], datum[-10:]))
                label = np.concatenate(([0 for _ in range(10)],[1 for _ in range(10)], [2 for _ in range(10)]))
            else:
                label = np.concatenate(([0 for _ in range(10)],[1 for _ in range(10)], [2 for _ in range(10)], [3 for _ in range(10)], [4 for _ in range(10)]))
            data.append(datum)
            labels.append(label)            
    groups = []
    if cross:
        for i in range(len(names)):
            for _ in range(data[i].shape[0] * len(samples[i])):
                groups.append(i)
    else:
        if classes == 2:
            num_per = 20
        elif classes == 3:
            num_per = 30 
        else:
            num_per = 50
        for i in range(len(data)):
            for _ in range(num_per):
                groups.append(i)
    groups = np.array(groups)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    return data, labels, groups
    

def fit(data,labels,groups,selected_channels,desired_features,model_path,scaler_path):
    wrapper = FeatureWrapper()
    processed_data = []
    for i, sample in enumerate(data):
        features = wrapper.compute_features(sample,i,125,selected_channels,desired_features=desired_features)
        processed_data.append(features)
    processed_data = np.array(processed_data)
    processed_data = np.reshape(processed_data,(processed_data.shape[0],-1))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(processed_data)
    processed_data = scaler.transform(processed_data)
    model = SVC(kernel="linear")
    model.fit(processed_data,labels)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
if __name__ == "__main__":
    names = ["onno", "yoyo", "emma"]
    samples = [[1,2,3,4,5,7,8],[1,2,3,4,5,6,7,8,9],[1,2]]
    data, labels, groups = get_dataset(names,samples,classes=3,cross=False)
    print(data.shape)
    print(labels.shape)
    print(groups.shape)
    selected_channels = [i for i in range(16)]
    desired_features = ["delta_bandpower","alpha_bandpower", "beta_bandpower", "theta_bandpower"]
    model_path = "models/three_class_model.joblib"
    scaler_path = "models/three_class_scaler.joblib"
    fit(data,labels,groups,selected_channels,desired_features,model_path,scaler_path)