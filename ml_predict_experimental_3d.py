# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:21:20 2024

@author: okovt
"""

#%%

import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit


def compute_fingerprints(traj):
    """Compute fingerprints for a single trajectory."""
    traj = np.array(traj)  
    positions = traj[:, :3]  
    times = traj[:, 3]  

    step_lengths = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    time_intervals = np.diff(times)
    velocities = step_lengths / time_intervals

    displacements = positions - positions[0]
    radii = np.linalg.norm(displacements, axis=1)

    center_of_mass = np.mean(positions, axis=0)
    radius_of_gyration = np.sqrt(
        np.mean(np.sum((positions - center_of_mass)**2, axis=1))
    )

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
        params, _ = curve_fit(power_law_3D, time_lags, msd_list, bounds=(0, [np.inf, 2]))
        D_fit, alpha = params
    except RuntimeError:
        D_fit, alpha = np.nan, np.nan

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
    }


def main():

    print("Loading trained model, scaler, and label encoder...")
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    print("Loading experimental trajectories...")
    trajectories_df = pd.read_csv("golgi7_spots.csv")

    print("Processing trajectories...")
    track_ids = trajectories_df["Track ID"].unique()
    fingerprints_list = []

    for track_id in track_ids:
        traj_df = trajectories_df[trajectories_df["Track ID"] == track_id].sort_values("T")
        if len(traj_df) < 2:
            print(f"Skipping Track_ID {track_id}: insufficient points.")
            continue

        traj = traj_df[["X", "Y", "Z", "T"]].values
        fingerprints = compute_fingerprints(traj)
        fingerprints["Track ID"] = track_id
        fingerprints_list.append(fingerprints)

    fingerprints_df = pd.DataFrame(fingerprints_list)

    feature_columns = [col for col in fingerprints_df.columns if col not in ["Track ID"]]
    X = fingerprints_df[feature_columns].values

    print("Scaling features...")
    X_scaled = scaler.transform(X)

    print("Predicting diffusion types...")
    predictions = model.predict(X_scaled)
    predicted_classes = label_encoder.inverse_transform(predictions)

    fingerprints_df["Predicted_Diffusion_Type"] = predicted_classes
    fingerprints_df.to_csv("predicted_experimental_results.csv", index=False)

    print("\nPredictions saved as 'predicted_experimental_results.csv'.")
    print("\nClass Distribution:")
    print(fingerprints_df["Predicted_Diffusion_Type"].value_counts())


if __name__ == "__main__":
    main()

#%%
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

print("Loading experimental trajectories...")
trajectories_df = pd.read_csv("golgi7_spots.csv")  

labels = pd.read_csv("predicted_experimental_results.csv")  

fig = plt.figure(figsize=(12, 12), dpi=600)
ax = fig.add_subplot(111, projection='3d')

colors = {"Normal": "green", "Directed": "orange", "Confined": "magenta", "Anomalous": "blue"}

for diffusion_type, color in colors.items():
    trajectories = labels[labels["Predicted_Diffusion_Type"] == diffusion_type]
    track_ids = trajectories["Track ID"].unique()
    for track_id in track_ids:
        track = trajectories_df[trajectories_df["Track ID"] == track_id][["X", "Y", "Z"]].values
        ax.plot(track[:, 0], track[:, 1], track[:, 2], color=color, alpha=0.6, label=diffusion_type)

handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), loc="best", fontsize=18)

ax.set_xlabel("X (μm)", fontsize=18, labelpad=10)
ax.set_ylabel("Y (μm)", fontsize=18, labelpad=10)
ax.set_zlabel("Z (μm)", fontsize=18)

ax.tick_params(axis='both', which='major', labelsize=16)


x_limits = [trajectories_df["X"].min(), trajectories_df["X"].max()]
y_limits = [trajectories_df["Y"].min(), trajectories_df["Y"].max()]
z_limits = [trajectories_df["Z"].min(), trajectories_df["Z"].max()]

max_range = max(
    x_limits[1] - x_limits[0], 
    y_limits[1] - y_limits[0], 
    z_limits[1] - z_limits[0]
) / 2.0

x_mid = (x_limits[1] + x_limits[0]) / 2.0
y_mid = (y_limits[1] + y_limits[0]) / 2.0
z_mid = (z_limits[1] + z_limits[0]) / 2.0

ax.set_xlim(x_mid - max_range, x_mid + max_range)
ax.set_ylim(y_mid - max_range, y_mid + max_range)
ax.set_zlim(z_mid - max_range, z_mid + max_range)

plt.tight_layout()
plt.show()

