# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:59:18 2024

@author: okovt
"""

#%%
import pickle
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit


def compute_fingerprints(traj):
    """Compute fingerprints for a single trajectory."""
    traj = np.array(traj)  
    positions = traj[:, :3]  
    times = np.arange(len(positions))  

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

    print("Loading synthetic trajectories from X_3D.pkl...")
    with open("X_3D.pkl", "rb") as f:
        trajectories_dict = pickle.load(f)

    print("Inspecting data format...")
    assert isinstance(trajectories_dict, dict), "Expected a dictionary of trajectories."
    print(f"Keys in the dictionary: {list(trajectories_dict.keys())}")

    fingerprints_list = []

    for diffusion_type, trajectories in trajectories_dict.items():
        print(f"Processing {diffusion_type} with {len(trajectories)} trajectories...")
        for i, traj in enumerate(trajectories):
            print(f"  Processing trajectory {i + 1}/{len(trajectories)} for {diffusion_type}...")
            fingerprints = compute_fingerprints(traj)
            fingerprints["Trajectory_ID"] = f"{diffusion_type}_{i}"
            fingerprints["Diffusion_Type"] = diffusion_type
            fingerprints_list.append(fingerprints)

    fingerprints_df = pd.DataFrame(fingerprints_list)

    print("Saving fingerprints to synthetic_fingerprints.csv...")
    fingerprints_df.to_csv("synthetic_fingerprints.csv", index=False)

    print("Fingerprints saved successfully!")

if __name__ == "__main__":
    main()

#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

print("Loading fingerprints data from CSV...")
features_df = pd.read_csv("synthetic_fingerprints.csv")

normal_trace_features = features_df[features_df['Diffusion_Type']=='Normal'].iloc[7]
directed_trace_features = features_df[features_df['Diffusion_Type']=='Directed'].iloc[10]
confined_trace_features = features_df[features_df['Diffusion_Type']=='Confined'].iloc[5]
anomalous_trace_features = features_df[features_df['Diffusion_Type']=='Anomalous'].iloc[20]

selected_features = pd.DataFrame([
    normal_trace_features,
    directed_trace_features,
    confined_trace_features,
    anomalous_trace_features
])

selected_features = selected_features.drop(columns=["Trajectory_ID", "Diffusion_Type"])

scaler = StandardScaler()
normalized_features = pd.DataFrame(
    scaler.fit_transform(selected_features),
    columns=selected_features.columns,
    index=["ND", "DD", "CD", "AD"]
)

plt.figure(figsize=(10, 6), dpi=600)
sns.heatmap(
    normalized_features, annot=True, cmap="coolwarm", cbar=True, linewidths=0.5
)

plt.xlabel("Features")
plt.ylabel("Diffusion Type")
plt.tight_layout()
plt.show()