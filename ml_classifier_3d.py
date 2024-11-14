# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:54:50 2024

@author: okovt
"""

#%%
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_test, y_pred, class_names):
    """Generate and visualize the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6),dpi=600)
    ax.matshow(cm, cmap="Blues")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
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
    plt.show()


def main():

    print("Loading fingerprints data from CSV...")
    fingerprints_df = pd.read_csv("synthetic_fingerprints.csv")

    print("Extracting features and target labels...")
    feature_columns = [col for col in fingerprints_df.columns if col not in ["Trajectory_ID", "Diffusion_Type"]]
    X = fingerprints_df[feature_columns].values
    y = fingerprints_df["Diffusion_Type"].values

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=7, stratify=y_encoded
    )

    print("Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.1,
        random_state=7,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    print("Saving model, scaler, and label encoder...")
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("Model, scaler, and label encoder saved successfully.")

    print("Evaluating the model...")
    y_pred = model.predict(X_test)

    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    class_names = label_encoder.classes_
    plot_confusion_matrix(
        y_test_decoded,
        y_pred_decoded,
        class_names=class_names,
        
    )

    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded, target_names=class_names))

if __name__ == "__main__":
    main()




