from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

### Method to classify a dataset with cross validation and compute relevant metrics using an sklearn model ###
def classify_sklearn(X, y, model, cv_splitter = StratifiedKFold(n_splits=5,shuffle=True), return_preds = False):
    outputs = []
    differences = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    X_reshaped = np.reshape(X,(X.shape[0],-1))
    for train, test in cv_splitter.split(X_reshaped,y):
        model.fit(X_reshaped[train],y[train])
        y_pred = model.predict(X_reshaped[test])
        differences.extend([abs(pred-real) for pred, real in zip(y_pred,y[test])])
        accuracies.append(accuracy_score(y[test],y_pred))
        precisions.append(precision_score(y[test],y_pred,average='weighted'))
        recalls.append(recall_score(y[test],y_pred,average='weighted'))
        f1_scores.append(f1_score(y[test],y_pred,average='weighted'))
        outputs.extend([(pred,real,index) for pred, real, index in zip(y_pred,y[test],test)])
    outputs = sorted(outputs, key=lambda x: x[2])
    outputs = [(pred,real) for pred, real, _ in outputs]
    mean_cv_accuracy = np.mean(accuracies)
    best_fold_accuracy = np.max(accuracies)
    worst_fold_accuracy = np.min(accuracies)
    mean_cv_precision = np.mean(precisions)
    best_fold_precision = np.max(precisions)
    worst_fold_precision = np.min(precisions)
    mean_cv_recall = np.mean(recalls)
    best_fold_recall = np.max(recalls)
    worst_fold_recall = np.min(recalls)
    mean_cv_f1 = np.mean(f1_scores)
    best_fold_f1 = np.max(f1_scores)
    worst_fold_f1 = np.min(f1_scores)
    mean_cv_difference = np.mean(differences)
    median_cv_difference = np.median(differences)
    metrics_dict = {
        'mean_accuracy': mean_cv_accuracy,
        'best_accuracy': best_fold_accuracy,
        'worst_accuracy': worst_fold_accuracy,
        'mean_precision': mean_cv_precision,
        'best_precision': best_fold_precision,
        'worst_precision': worst_fold_precision,
        'mean_recall': mean_cv_recall,
        'best_recall': best_fold_recall,
        'worst_recall': worst_fold_recall,
        'mean_f1': mean_cv_f1,
        'best_f1': best_fold_f1,
        'worst_f1': worst_fold_f1,
        'mean_difference': mean_cv_difference,
        'median_difference': median_cv_difference
    }
    if return_preds:
        metrics_dict['predictions'] = outputs
    return metrics_dict
    
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

### Method to classify a dataset with cross validation and compute relevant metrics using a pytorch model ###
def classify_torch(X, y, model, cv_splitter = StratifiedKFold(n_splits=5, shuffle=True), return_preds = False, batch_size = 10, learning_rate = 0.01, num_epochs = 10, criterion = nn.CrossEntropyLoss()):
    predictions = []
    differences = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    data_tensor = torch.tensor(np.expand_dims(X,axis=2),dtype=torch.float32)
    labels_tensor = torch.tensor(y,dtype=torch.long)
    for train, test in cv_splitter.split(data_tensor,labels_tensor):
        reset_weights(model)
        train_data, test_data = data_tensor[train], data_tensor[test]
        train_labels, test_labels = labels_tensor[train], labels_tensor[test]
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        with torch.no_grad():
            curr_idx = 0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                differences.extend([abs(pred-real) for pred, real in zip(preds,targets)])
                accuracies.append(accuracy_score(targets,preds))
                precisions.append(precision_score(targets,preds,average='weighted'))
                recalls.append(recall_score(targets,preds,average='weighted'))
                f1_scores.append(f1_score(targets,preds,average='weighted'))
                predictions.extend([(pred,real,index) for pred, real, index in zip(preds,targets,test[curr_idx:curr_idx+len(targets)])])
                curr_idx = len(targets)
    predictions = sorted(predictions, key=lambda x: x[2])
    predictions = [(pred,real) for pred, real, _ in predictions]
    mean_cv_accuracy = np.mean(accuracies)
    best_fold_accuracy = np.max(accuracies)
    worst_fold_accuracy = np.min(accuracies)
    mean_cv_precision = np.mean(precisions)
    best_fold_precision = np.max(precisions)
    worst_fold_precision = np.min(precisions)
    mean_cv_recall = np.mean(recalls)
    best_fold_recall = np.max(recalls)
    worst_fold_recall = np.min(recalls)
    mean_cv_f1 = np.mean(f1_scores)
    best_fold_f1 = np.max(f1_scores)
    worst_fold_f1 = np.min(f1_scores)
    mean_cv_difference = np.mean(differences)
    median_cv_difference = np.median(differences)
    metrics_dict = {
        'mean_accuracy': mean_cv_accuracy,
        'best_accuracy': best_fold_accuracy,
        'worst_accuracy': worst_fold_accuracy,
        'mean_precision': mean_cv_precision,
        'best_precision': best_fold_precision,
        'worst_precision': worst_fold_precision,
        'mean_recall': mean_cv_recall,
        'best_recall': best_fold_recall,
        'worst_recall': worst_fold_recall,
        'mean_f1': mean_cv_f1,
        'best_f1': best_fold_f1,
        'worst_f1': worst_fold_f1,
        'mean_difference': mean_cv_difference,
        'median_difference': median_cv_difference
    }
    if return_preds:
        metrics_dict['predictions'] = predictions
    return metrics_dict