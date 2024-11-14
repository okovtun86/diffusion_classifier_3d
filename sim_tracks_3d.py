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
    Generate random parameters for 3D diffusion types based on Kowalek et al. (2020).
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
    v = np.sqrt(R * 4 * D / TDM)

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
    numparams = 200  
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("Loading synthetic trajectories from X_3D.pkl...")
with open("X_3D.pkl", "rb") as f:
    traces = pickle.load(f)

normal_trace_sample = traces["Normal"][7]
directed_trace_sample = traces["Directed"][10]
confined_trace_sample = traces["Confined"][5]
anomalous_trace_sample = traces["Anomalous"][20]

all_trajectories = [normal_trace_sample, directed_trace_sample, confined_trace_sample, anomalous_trace_sample]
all_points = [point for traj in all_trajectories for point in traj]
all_points = np.array(all_points)
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

fig = plt.figure(figsize=(10, 8), dpi=600)
ax = fig.add_subplot(111, projection='3d')

ax.plot(normal_trace_sample[:, 0], normal_trace_sample[:, 1], normal_trace_sample[:, 2], color='green', label='Normal Diffusion')
ax.plot(directed_trace_sample[:, 0], directed_trace_sample[:, 1], directed_trace_sample[:, 2], color='orange', label='Directed Motion')
ax.plot(confined_trace_sample[:, 0], confined_trace_sample[:, 1], confined_trace_sample[:, 2], color='magenta', label='Confined Diffusion')
ax.plot(anomalous_trace_sample[:, 0], anomalous_trace_sample[:, 1], anomalous_trace_sample[:, 2], color='blue', label='Anomalous Diffusion')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

ax.legend()

plt.tight_layout()
plt.show()

#%%

import pickle

file_path = 'X_3D.pkl'

with open(file_path, 'wb') as file:
    pickle.dump(traces, file)

print(f"Traces saved successfully as {file_path}")