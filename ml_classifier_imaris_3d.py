# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 21:34:07 2025

@author: okovt
"""
#%%
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

base_folder = 'imaris_sim_csv'  
motion_types = ['anomalous_all_stats', 'confined_all_stats','directed_all_stats','normal_all_stats']
motion_labels = ['Anomalous', 'Confined', 'Directed', 'Normal']

data = pd.DataFrame()

for motion_type, motion_label in zip(motion_types, motion_labels):
    folder_path = os.path.join(base_folder, motion_type)
    motion_data = pd.DataFrame()
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            
            feature_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path, skiprows=3)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
            
            if df.columns[0] != feature_name:
                df.rename(columns={df.columns[0]: feature_name}, inplace=True)
            
            df = df[[feature_name]]
            
            if motion_data.empty:
                motion_data = df
            else:
                motion_data = pd.concat([motion_data, df], axis=1)
    
    motion_data['Label'] = motion_label
    data = pd.concat([data, motion_data], ignore_index=True)

data['Label'] = data['Label'].astype('category')
data['Label'] = data['Label'].cat.codes
label_mapping = {i: label for i, label in enumerate(motion_labels)}

X = data.drop(columns=['Label'])
y = data['Label']

X = X.fillna(X.mean())

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = xgb.XGBClassifier(
    use_label_encoder=False, n_estimators=1000,
    max_depth=5,
    learning_rate=0.1,
    random_state=7,
    eval_metric="mlogloss"
    )

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=list(label_mapping.keys())))

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=[label_mapping[i] for i in sorted(label_mapping.keys())]))

#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_normalized_feature_map(X, labels, feature_names):
    """Plot normalized feature map as a heatmap."""
    mean_values = []
    for label in np.unique(labels):
        mean_values.append(np.mean(X[labels == label, :], axis=0))

    mean_values = np.array(mean_values)

    plt.figure(figsize=(12, 6), dpi=600)
    sns.heatmap(
        mean_values,
        annot=True,
        annot_kws={"size": 12}, 
        cmap="coolwarm",
        cbar=True,
        cbar_kws={"shrink": 0.8, "aspect": 10, "format": "%.1f"},  
        linewidths=0.5,
        fmt=".2f",  
        xticklabels=feature_names,
        yticklabels=["AD", "CD", "DM", "ND"]  
    )

    plt.xlabel("Normalized Imaris Features", fontsize=18)
    plt.ylabel("Diffusion Type", fontsize=18)
    plt.xticks(fontsize=16, rotation=45, ha="right")
    plt.yticks(fontsize=16)
  
    output_file = "normalized_imaris_feature_map.png"
    print(f"Normalized Imaris feature map saved as '{output_file}'.")
    plt.show()

plot_normalized_feature_map(X_train.values, y_train.values, X.columns.tolist())




#%%

import joblib
joblib.dump(clf, 'xgboost_imaris_model.pkl')
joblib.dump(scaler, 'standard_scaler_imaris.pkl')

print("Model training and evaluation complete.")

#%%
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(y_test, y_pred, class_names):
    """Generate and visualize the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=600)
    ax.matshow(cm, cmap="Blues")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > np.max(cm) / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", color=color)

    ax.set(
        yticks=range(len(class_names)),
        xticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
    )
    ax.xaxis.set_ticks_position("bottom")
    plt.tight_layout()

    output_file = "confusion_matrix_imaris_classifier.png"
    plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight')
    print(f"Figure saved as '{output_file}'.")

    plt.show()

plot_confusion_matrix(y_test, y_pred, [label_mapping[i] for i in sorted(label_mapping.keys())])

print("Model training, evaluation, and visualization complete.")

#%%
import pandas as pd
import os
import joblib

experimental_folder = 'experimental_all_stats'
experimental_data = pd.DataFrame()

for file_name in os.listdir(experimental_folder):
    if file_name.endswith('.csv'):
        feature_name = os.path.splitext(file_name)[0]
        file_path = os.path.join(experimental_folder, file_name)
        try:
            df = pd.read_csv(file_path, skiprows=3)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        if df.columns[0] != feature_name:
            df.rename(columns={df.columns[0]: feature_name}, inplace=True)

        df = df[[feature_name]]

        if experimental_data.empty:
            experimental_data = df
        else:
            experimental_data = pd.concat([experimental_data, df], axis=1)

scaler = joblib.load('standard_scaler_imaris.pkl')
experimental_data = experimental_data.fillna(experimental_data.mean())
experimental_data = pd.DataFrame(scaler.transform(experimental_data), columns=experimental_data.columns)

clf = joblib.load('xgboost_imaris_model.pkl')

experimental_predictions = clf.predict(experimental_data)
experimental_labels = [label_mapping[label] for label in experimental_predictions]

output_predictions = pd.DataFrame({'Trajectory': range(len(experimental_labels)), 'Predicted Motion Type': experimental_labels})
output_predictions.to_csv('experimental_predictions_normalized_imaris.csv', index=False)
print("Predictions saved to 'experimental_predictions_normalized_imaris.csv'.")

prediction_counts = pd.Series(experimental_labels).value_counts()
print("Prediction Counts by Motion Type:")
print(prediction_counts)

prediction_counts.to_csv('prediction_report_normalized_imaris.csv', header=['Count'], index_label='Motion Type')
print("Prediction report saved to 'prediction_report_normalized_imaris.csv'.")





