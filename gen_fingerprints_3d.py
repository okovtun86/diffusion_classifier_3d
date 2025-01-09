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
    dt = 1
    traj = np.array(traj)  
    positions = traj[:, :3]  
    times = np.arange(len(positions))  

    step_lengths = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    time_intervals = np.diff(times)*dt
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

    time_lags = (np.arange(1, max_lag))*dt

    def power_law_3D(t, D, alpha):
        return 6 * D * t**alpha

    try:
        params, pcov = curve_fit(power_law_3D, time_lags, msd_list, bounds=(0, [np.inf, 2]))
        D_fit, alpha = params
        msdpred = power_law_3D(time_lags, *params)
        
        # residual sum of squares
        ss_res = np.sum((msd_list - msdpred) ** 2)
        # total sum of squares
        ss_tot = np.sum((msd_list - np.mean(msd_list)) ** 2)

        # r-squared
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


def main():

    print("Loading synthetic trajectories from X_3D_v2.pkl...")
    with open("X_3D_v2.pkl", "rb") as f:
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

    print("Saving fingerprints to synthetic_fingerprints_v2.csv...")
    fingerprints_df.to_csv("synthetic_fingerprints_dt_1.csv", index=False)

    print("Fingerprints saved successfully!")

if __name__ == "__main__":
    main()

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


print("Loading fingerprints data from CSV...")
features_df = pd.read_csv("synthetic_fingerprints_v2.csv")

average_features = features_df.groupby("Diffusion_Type").mean()

average_features = average_features.drop(columns=["Trajectory_ID"], errors="ignore")

scaler = StandardScaler()
normalized_features = pd.DataFrame(
    scaler.fit_transform(average_features),
    columns=average_features.columns,
    index=["AD", "CD", "DM", "ND"]  
)


plt.figure(figsize=(12, 8), dpi=600)
sns.heatmap(
    normalized_features,
    annot=True,
    annot_kws={"size": 14},  
    cmap="coolwarm",
    cbar=True,
    cbar_kws={"shrink": 0.8, "aspect": 10, "format": "%.1f"},  
    linewidths=0.5,
    fmt=".2f"  
)

plt.xlabel("Normalized Features", fontsize=18)
plt.ylabel("Diffusion Type", fontsize=18)
plt.xticks(fontsize=16, rotation=45, ha="right")  
plt.yticks(fontsize=16)


colorbar = plt.gca().collections[0].colorbar
colorbar.ax.tick_params(labelsize=16)  # Adjust tick font size

plt.tight_layout()

output_file = "fingerprints_dt_1.png"
plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight')
print(f"Figure saved as '{output_file}'.")

plt.show()
