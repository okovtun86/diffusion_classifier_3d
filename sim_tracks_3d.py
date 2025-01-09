# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:05:49 2024

@author: okovt
"""

#%%

import numpy as np
from fbm import fgn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def GenerateParams3D(numparams, dt, D):
    """
    Generate random parameters for 3D diffusion types based on work by
    Wagner et al., Kowalek et al., and Pinholt et al. 
    
    """
    Nmin, Nmax = 30, 600
    Bmin, Bmax = 1, 6
    Rmin, Rmax = 1, 17
    alphamin, alphamax = 0.3, 0.7
    Qmin, Qmax = 1, 9

    Q = np.random.uniform(Qmin, Qmax, size=numparams)
    Q1, Q2 = Q, Q

    NsND = np.random.randint(Nmin, Nmax + 1, size=numparams)
    NsAD = np.random.randint(Nmin, Nmax + 1, size=numparams)
    NsCD = np.random.randint(Nmin, Nmax + 1, size=numparams)
    NsDM = np.random.randint(Nmin, Nmax + 1, size=numparams)
    TDM = NsDM * dt

    B = np.random.uniform(Bmin, Bmax, size=numparams)
    r_c = np.sqrt(D * NsCD * dt / B)

    R = np.random.uniform(Rmin, Rmax, size=numparams)
    v = np.sqrt(R * 6 * D / TDM)

    alpha = np.random.uniform(alphamin, alphamax, size=numparams)

    sigmaND = np.sqrt(D * dt) / Q1
    sigmaAD = np.sqrt(D * dt) / Q1
    sigmaCD = np.sqrt(D * dt) / Q1
    sigmaDM = np.sqrt(D * dt + v**2 * dt**2) / Q2

    return {
        "NsND": NsND,
        "NsAD": NsAD,
        "NsCD": NsCD,
        "NsDM": NsDM,
        "D": D,
        "dt": dt,
        "r_c": r_c,
        "v": v,
        "alpha": alpha,
        "sigmaND": sigmaND,
        "sigmaAD": sigmaAD,
        "sigmaCD": sigmaCD,
        "sigmaDM": sigmaDM,
    }


def Gen_normal_diff_3D(D, dt, sigma1s, Ns, withlocerr=True):
    """Generate normal diffusion traces in 3D."""
    traces = []
    for n, sig in zip(Ns, sigma1s):
        xsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
        ysteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
        zsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
        x, y, z = (
            np.concatenate([[0], np.cumsum(xsteps)]),
            np.concatenate([[0], np.cumsum(ysteps)]),
            np.concatenate([[0], np.cumsum(zsteps)]),
        )
        if withlocerr:
            x_noisy, y_noisy, z_noisy = (
                x + np.random.normal(0, sig, size=x.shape),
                y + np.random.normal(0, sig, size=y.shape),
                z + np.random.normal(0, sig, size=z.shape),
            )
            traces.append(np.array([x_noisy, y_noisy, z_noisy]).T)
        else:
            traces.append(np.array([x, y, z]).T)
    return traces


def Gen_directed_diff_3D(D, dt, vs, sigmaDM, Ns, beta_set=None, gamma_set=None, withlocerr=True):
    """Generate directed motion traces in 3D."""
    traces = []
    for v, n, sig in zip(vs, Ns, sigmaDM):
        if beta_set is None:
            beta = np.random.uniform(0, 2 * np.pi)
        else:
            beta = beta_set
        if gamma_set is None:
            gamma = np.random.uniform(0, np.pi)
        else:
            gamma = gamma_set

        dx, dy, dz = (
            v * dt * np.sin(gamma) * np.cos(beta),
            v * dt * np.sin(gamma) * np.sin(beta),
            v * dt * np.cos(gamma),
        )

        xsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dx
        ysteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dy
        zsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dz

        x, y, z = (
            np.concatenate([[0], np.cumsum(xsteps)]),
            np.concatenate([[0], np.cumsum(ysteps)]),
            np.concatenate([[0], np.cumsum(zsteps)]),
        )
        if withlocerr:
            x_noisy, y_noisy, z_noisy = (
                x + np.random.normal(0, sig, size=x.shape),
                y + np.random.normal(0, sig, size=y.shape),
                z + np.random.normal(0, sig, size=z.shape),
            )
            traces.append(np.array([x_noisy, y_noisy, z_noisy]).T)
        else:
            traces.append(np.array([x, y, z]).T)
    return traces


def Gen_confined_diff_3D(D, dt, r_cs, sigmaCD, Ns, withlocerr=True, nsubsteps=100):
    """Generate confined diffusion traces in 3D."""
    def get_confined_step(x0, y0, z0, D, dt, r_c):
        dt_prim = dt / nsubsteps
        for _ in range(nsubsteps):
            x1, y1, z1 = (
                x0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
                y0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
                z0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
            )
            if np.sqrt(x1**2 + y1**2 + z1**2) <= r_c:
                x0, y0, z0 = x1, y1, z1
        return x1, y1, z1

    traces = []
    for r_c, sig, n in zip(r_cs, sigmaCD, Ns):
        xs, ys, zs = [], [], []
        x0, y0, z0 = 0, 0, 0
        for _ in range(n + 1):
            xs.append(x0)
            ys.append(y0)
            zs.append(z0)
            x0, y0, z0 = get_confined_step(x0, y0, z0, D, dt, r_c)
        x, y, z = np.array(xs), np.array(ys), np.array(zs)
        if withlocerr:
            x_noisy, y_noisy, z_noisy = (
                x + np.random.normal(0, sig, size=x.shape),
                y + np.random.normal(0, sig, size=y.shape),
                z + np.random.normal(0, sig, size=z.shape),
            )
            traces.append(np.array([x_noisy, y_noisy, z_noisy]).T)
        else:
            traces.append(np.array([x, y, z]).T)
    return traces


def Gen_anomalous_diff_3D(D, dt, alphs, sigmaAD, Ns, withlocerr=True):
    """Generate anomalous diffusion traces in 3D."""
    Hs = alphs / 2
    traces = []
    for n, sig, H in zip(Ns, sigmaAD, Hs):
        n = int(n)
        stepx, stepy, stepz = (
            np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte"),
            np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte"),
            np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte"),
        )
        x, y, z = (
            np.concatenate([[0], np.cumsum(stepx)]),
            np.concatenate([[0], np.cumsum(stepy)]),
            np.concatenate([[0], np.cumsum(stepz)]),
        )
        if withlocerr:
            x_noisy, y_noisy, z_noisy = (
                x + np.random.normal(0, sig, size=x.shape),
                y + np.random.normal(0, sig, size=y.shape),
                z + np.random.normal(0, sig, size=z.shape),
            )
            traces.append(np.array([x_noisy, y_noisy, z_noisy]).T)
        else:
            traces.append(np.array([x, y, z]).T)
    return traces


def main():
    numparams = 500  
    dt = 1       
    D = 0.02          

    params = GenerateParams3D(numparams, dt, D)

    print("Generating Normal Diffusion Trajectories...")
    normal_traces = Gen_normal_diff_3D(
        D=params["D"], dt=params["dt"], sigma1s=params["sigmaND"], Ns=params["NsND"]
    )

    print("Generating Directed Motion Trajectories...")
    directed_traces = Gen_directed_diff_3D(
        D=params["D"], dt=params["dt"], vs=params["v"], sigmaDM=params["sigmaDM"], Ns=params["NsDM"]
    )

    print("Generating Confined Diffusion Trajectories...")
    confined_traces = Gen_confined_diff_3D(
        D=params["D"], dt=params["dt"], r_cs=params["r_c"], sigmaCD=params["sigmaCD"], Ns=params["NsCD"]
    )

    print("Generating Anomalous Diffusion Trajectories...")
    anomalous_traces = Gen_anomalous_diff_3D(
        D=params["D"], dt=params["dt"], alphs=params["alpha"], sigmaAD=params["sigmaAD"], Ns=params["NsAD"]
    )

    all_traces = {
        "Normal": normal_traces,
        "Directed": directed_traces,
        "Confined": confined_traces,
        "Anomalous": anomalous_traces,
    }

    return all_traces


traces = main()


#%%

import pickle

file_path = 'X_3D_v2.pkl'

with open(file_path, 'wb') as file:
    pickle.dump(traces, file)

print(f"Traces saved successfully as {file_path}")



#%%
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load synthetic trajectories
print("Loading synthetic trajectories from X_3D_v2.pkl...")
with open("X_3D_v2.pkl", "rb") as f:
    traces = pickle.load(f)

# Define diffusion types, colors, and offsets
diffusion_types = ["Normal", "Directed", "Confined", "Anomalous"]
colors = {"Normal": "green", "Directed": "orange", "Confined": "magenta", "Anomalous": "blue"}
offsets = {"Normal": (0, 0, 0), "Directed": (50, 0, 0), "Confined": (0, 50, 0), "Anomalous": (50, 50, 0)}

# Calculate axis limits based on all trajectories
all_points = np.vstack([np.vstack(traces[diff_type]) for diff_type in diffusion_types])
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

# Normalize axis limits
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2
z_center = (z_max + z_min) / 2

x_min, x_max = x_center - max_range / 2, x_center + max_range / 2
y_min, y_max = y_center - max_range / 2, y_center + max_range / 2
z_min, z_max = z_center - max_range / 2, z_center + max_range / 2

# Create a single 3D plot for all trajectories
fig = plt.figure(figsize=(10, 8), dpi=600)
ax = fig.add_subplot(111, projection='3d')

for diff_type in diffusion_types:
    for trajectory in traces[diff_type]:
        offset = offsets[diff_type]
        trajectory_offset = trajectory + np.array(offset)
        ax.plot(
            trajectory_offset[:, 0], trajectory_offset[:, 1], trajectory_offset[:, 2],
            color=colors[diff_type], alpha=0.6, linewidth=0.8
        )

# Set normalized axis limits
ax.set_xlim(x_min, x_max + 50)
ax.set_ylim(y_min, y_max + 50)
ax.set_zlim(z_min, z_max)

# Add ticks and increase tick font size
ax.tick_params(axis='both', which='major', labelsize=16)
ax.xaxis.set_tick_params(width=2, length=6)
ax.yaxis.set_tick_params(width=2, length=6)
ax.zaxis.set_tick_params(width=2, length=6)

# Set axis labels with increased font size
ax.set_xlabel("X (μm)", labelpad=17, fontsize=18)
ax.set_ylabel("Y (μm)", labelpad=17, fontsize=18)
ax.set_zlabel("Z (μm)", labelpad=17, fontsize=18)

# Customize legend with horizontal layout
legend_labels = [
    plt.Line2D([0], [0], color=colors[diff_type], lw=2, label=diff_type)
    for diff_type in diffusion_types
]
ax.legend(
    handles=legend_labels,
    loc="upper center",        # Position at the top center of the plot
    bbox_to_anchor=(0.5, 1.01), # Adjust location outside the plot
    ncol=len(diffusion_types), # Arrange labels in a single row
    fontsize=18               # Increase font size
)

# Final adjustments and show plot
plt.tight_layout()


# Save the figure as a PNG file
output_file = "3D_diffusion_trajectories.png"
plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight')
print(f"Figure saved as '{output_file}'.")

plt.show()



