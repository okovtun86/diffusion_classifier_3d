# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:43:37 2025

@author: okovt
"""

#%%
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from collections import defaultdict
from scipy.stats import mode

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def compute_fingerprints(traj):
    """Compute fingerprints for a single trajectory segment."""
    
    traj = np.array(traj)  
    positions = traj[:, :3]  
    times = np.arange(len(positions))  

    step_lengths = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    time_intervals = np.diff(times)
    velocities = step_lengths / time_intervals

    displacements = positions - positions[0]
    radii = np.linalg.norm(displacements, axis=1)

    center_of_mass = np.mean(positions, axis=0)
    radius_of_gyration = np.sqrt(np.mean(np.sum((positions - center_of_mass)**2, axis=1)))

    total_path_length = np.sum(step_lengths)
    end_to_start_distance = np.linalg.norm(displacements[-1])
    straightness_index = end_to_start_distance / total_path_length

    msd_list = []
    max_lag = int(0.25 * len(times))  
    for lag in range(1, max_lag):
        squared_displacements = [
            np.linalg.norm(positions[j] - positions[j + lag])**2
            for j in range(len(positions) - lag)
        ]
        msd_list.append(np.mean(squared_displacements))

    time_lags = np.arange(1, max_lag)

    def power_law_3D(t, D, alpha):
        return 6 * D * t**alpha

    try:
        params, pcov = curve_fit(power_law_3D, time_lags, msd_list, bounds=(0, [np.inf, 2]))
        D_fit, alpha = params
        msdpred = power_law_3D(time_lags, *params)
        
        ss_res = np.sum((msd_list - msdpred) ** 2)
        ss_tot = np.sum((msd_list - np.mean(msd_list)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
    except RuntimeError:
        D_fit, alpha, r2 = np.nan, np.nan, np.nan

    max_radius = np.max(radii)
    range_of_radii = np.max(radii) - np.min(radii)
    trappedness = 1.0 - max_radius / (np.sum(radii) / len(radii))
    gaussianity = np.mean((step_lengths - np.mean(step_lengths))**4) / (np.var(step_lengths)**2)

    skewness_step_length = skew(step_lengths)
    kurtosis_step_length = kurtosis(step_lengths)
    mean_velocity = np.mean(velocities)
    std_velocity = np.std(velocities)

    return {
        "Max_Radius": max_radius,
        "Range_of_Radii": range_of_radii,
        "Trappedness": trappedness,
        "Gaussianity": gaussianity,
        "Skewness_Step_Length": skewness_step_length,
        "Kurtosis_Step_Length": kurtosis_step_length,
        "Mean_Velocity": mean_velocity,
        "Std_Velocity": std_velocity,
        "Radius_of_Gyration": radius_of_gyration,
        "Straightness_Index": straightness_index,
        "D_PowerLaw": D_fit,  
        "Alpha_PowerLaw": alpha,
        "Goodness_of_fit": r2,
    }


def classify_with_sliding_window(trajectory, model, scaler, window_size, step_size):
    """
    Classify a single trajectory using a sliding window approach with majority voting for overlapping windows.

    Parameters:
        trajectory (np.ndarray): A 3D trajectory with shape (n_points, 4) [X, Y, Z, T].
        model (sklearn.Model): Pre-trained machine learning classifier.
        scaler (sklearn.preprocessing.StandardScaler): Scaler for normalizing features.
        window_size (int): Number of points in each sliding window.
        step_size (int): Step size for sliding the window.

    Returns:
        list: Predicted labels for each frame in the trajectory using majority voting for overlaps.
    """
    num_points = len(trajectory)
    frame_votes = defaultdict(list)  

    for start in range(0, num_points - window_size + 1, step_size):
        
        window = trajectory[start : start + window_size]

        features = compute_fingerprints(window)
        features_df = pd.DataFrame([features])  
        features_scaled = scaler.transform(features_df.values)  

        prediction = model.predict(features_scaled)[0]

        for i in range(start, start + window_size):
            frame_votes[i].append(prediction)

    full_predictions = []
    for i in range(num_points):
        if i in frame_votes:
            majority_label = mode(frame_votes[i]).mode[0]
            full_predictions.append(majority_label)
        else:
            full_predictions.append(None)

    return full_predictions



print("Loading trained model, scaler, and label encoder...")
with open("xgboost_model_dt_1.pkl", "rb") as f:
  model = pickle.load(f)
with open("scaler_dt_1.pkl", "rb") as f:
  scaler = pickle.load(f)
with open("label_encoder_dt_1.pkl", "rb") as f:
  label_encoder = pickle.load(f)

print("Loading mixed trajectories with ground truth labels...")
with open("mixed_binary_diffusion_tracks_CD_ND_3D.pkl", "rb") as f:
  mixed_trajectories, true_labels = pickle.load(f)

print("Classifying mixed trajectories with sliding window...")
window_size = 50
step_size = 25
all_predictions = []
flat_true_labels = []

for i, (trajectory, trajectory_labels) in enumerate(zip(mixed_trajectories, true_labels)):
    
        predictions = classify_with_sliding_window(trajectory, model, scaler, window_size, step_size)
        print(f"Trajectory {i}: Predictions length = {len(predictions)}, Expected length = {len(trajectory)}")

        all_predictions.extend(predictions)
        flat_true_labels.extend(trajectory_labels)

#%%

print("Generating confusion matrix...")
y_true = label_encoder.transform(flat_true_labels)
y_pred = np.array(all_predictions)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_trajectory_with_true_and_predicted_labels(trajectory, true_labels, predicted_labels, label_encoder):
    """
    Visualize a single 3D trajectory with true and predicted diffusion types.

    Parameters:
        trajectory (np.ndarray): The trajectory data with shape (n_points, 3) [X, Y, Z].
        true_labels (list): True labels for each frame in the trajectory.
        predicted_labels (list): Predicted labels for each frame in the trajectory.
        label_encoder (sklearn.preprocessing.LabelEncoder): Label encoder used for classification.
    """
    import matplotlib.pyplot as plt

    unique_labels = label_encoder.classes_
    color_map = {"Normal": "green", "Directed": "orange", "Confined": "magenta", "Anomalous": "blue"}
    true_colors = [color_map[true_labels[i]] for i in range(len(true_labels))]
    predicted_colors = [color_map[label_encoder.inverse_transform([pred])[0]] for pred in predicted_labels]

    fig = plt.figure(figsize=(16, 8), dpi=600)
 
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(len(trajectory) - 1):
        ax1.plot(
            trajectory[i : i + 2, 0],  
            trajectory[i : i + 2, 1],  
            trajectory[i : i + 2, 2],  
            color=true_colors[i],
            alpha=0.8,
            lw=2,
        )
    ax1.set_xlabel("X (\u03bcm)", fontsize=18, labelpad = 10)
    ax1.set_ylabel("Y (\u03bcm)", fontsize=18, labelpad = 10)
    ax1.set_zlabel("Z (\u03bcm)", fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(len(trajectory) - 1):
        ax2.plot(
            trajectory[i : i + 2, 0], 
            trajectory[i : i + 2, 1],  
            trajectory[i : i + 2, 2],  
            color=predicted_colors[i],
            alpha=0.8,
            lw=2,
        )
    ax2.set_xlabel("X (\u03bcm)", fontsize=18, labelpad = 10)
    ax2.set_ylabel("Y (\u03bcm)", fontsize=18, labelpad = 10)
    # ax2.set_zlabel("Z (\u03bcm)", fontsize=18, labelpad = 10)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    handles = [plt.Line2D([0], [0], color=color_map[label], lw=4) for label in unique_labels]
    fig.legend(
        handles,
        unique_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(unique_labels),
        fontsize=18,
        
    )

    plt.tight_layout() 

    output_file = "mixed_CD_ND_comparison.png"
    plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight')
    print(f"Figure saved as '{output_file}'.")    
    
    plt.show()


trajectory = mixed_trajectories[10][:, :3] 

current_true_labels = true_labels[10]  

predicted_labels = classify_with_sliding_window(
    trajectory, model, scaler, window_size=50, step_size=25
)

visualize_trajectory_with_true_and_predicted_labels(trajectory, current_true_labels, predicted_labels, label_encoder)
