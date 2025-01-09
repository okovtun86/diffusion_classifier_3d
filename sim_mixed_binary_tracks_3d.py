# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:31:26 2025

@author: okovt
"""

#%%
import numpy as np
from fbm import fgn
import matplotlib.pyplot as plt


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


def Get_shifting_diff(diff_gens, params, state, lens, plot=False):
    """Generates binary state-shifting diffusion between the two trace generators given
    in diff_gens.

    Parameters
    ----------
    diff_gens : list of length 2.
        Should contain two of the four possible trace-generators:
        - Gen_normal_diff_3D
        - Gen_directed_diff_3D
        - Gen_confined_diff_3D
        - Gen_anomalous_diff_3D.
    
    """
    diff1, diff2 = [diff_gens[i](**params[i]) for i in range(2)]
    if len(state) == 1 and state[0] == 0:
        trace = diff1[0][:-1]
    elif len(state) == 1 and state[0] == 1:
        trace = diff2[0][:-1]
    else:
        c0, c1 = 0, 0
        stepsx, stepsy, stepsz  = [], [], []
        for s in state:
            trace_choice = [diff1, diff2][s][[c0, c1][s]]
            step_comp = trace_choice[1:] - trace_choice[:-1]
            stepsx += list(step_comp[:, 0])
            stepsy += list(step_comp[:, 1])
            stepsz += list(step_comp[:, 2])
            # steps = np.concatenate([steps, step_comp])
            if s == 0:
                c0 += 1
            else:
                c1 += 1

        trace = np.concatenate(
            [
                np.array([[0, 0, 0]]),
                np.array([np.cumsum(stepsx), np.cumsum(stepsy), np.cumsum(stepsz)]).T[:-1],
            ]
        )
    if len(trace) != np.sum(lens):
        # print(diff1, diff2, trace, cropped_diff1, cropped_diff2)
        print(diff1, diff2, trace)  # , cropped_diff1, cropped_diff2)
        raise ValueError(
            f"{len(trace)},{state},{lens},{[len(i) for i in diff1],[len(i) for i in diff2]}"
        )
        # raise ValueError(
        #     f"{len(trace)},{state},{c0,c1},{lens},{[len(i) for i in diff1],[len(i) for i in diff2]},{[len(i) for i in cropped_diff1],[len(i) for i in cropped_diff2]}"
        # )
    trace = np.array(trace)
    if plot:
        SLS = np.sqrt(np.sum((trace[1:] - trace[:-1]) ** 2, axis=1))
        frames = np.arange(len(SLS))
        n = 0
        cols = ["dimgrey", "darkred"]
        
        fig = plt.figure(figsize=(12, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        for s, l in zip(state, lens):
            ax1.plot(
                trace[:, 0][n : n + l + 1], 
                trace[:, 1][n : n + l + 1],
                trace[:, 2][n : n + l + 1],
                c=cols[s]
            )
            ax2.plot(
                frames[n : np.min([n + l + 1, len(SLS)])],
                SLS[n : np.min([n + l + 1, len(SLS)])],
                "o",
                c=cols[s],
            )
            n += l
        
        ax1.set_title("3D Trajectory with State Changes")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax2.set_title("Step Lengths")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Step Length")
        plt.tight_layout()
        plt.show()
    return trace

#%%
import numpy as np
import pickle

def generate_mixed_diffusion_tracks(num_tracks, diff_gens, state, lens):
    """
    Generate mixed diffusion tracks with alternating diffusion modes.

    Parameters:
        num_tracks (int): Number of tracks to generate.
        diff_gens (list): List of diffusion generators.
        params (list): List of parameters for each diffusion mode.
        state (list): State sequence indicating diffusion type transitions.
        lens (list): Length of each segment for the states.

    Returns:
        list: List of generated tracks.
        list: List of associated diffusion type annotations (labels).
    """
    tracks = []
    labels = []
    
    D=0.02
    dt=1
    
    Bmin, Bmax = 1, 6
    Rmin, Rmax = 1, 17
    alphamin, alphamax = 0.3, 0.7
    Qmin, Qmax = 1, 9

    Q = np.random.uniform(Qmin, Qmax, size=6)
    Q1, Q2 = Q, Q

    NsND = np.array(lens[0:6])
    NsAD = np.array(lens[0:6])
    NsCD = np.array(lens[0:6])
    NsDM = np.array(lens[0:6])
    TDM = NsDM * dt

    alpha = np.random.uniform(alphamin, alphamax, size=6)

    for _ in range(num_tracks):
        
        B = np.random.uniform(Bmin, Bmax, size=6)
        r_c = np.sqrt(D * NsCD * dt / B)
        
        R = np.random.uniform(Rmin, Rmax, size=6)
        v = np.sqrt(R * 6 * D / TDM)
        
        sigmaND = np.sqrt(D * dt) / Q1
        sigmaAD = np.sqrt(D * dt) / Q1
        sigmaCD = np.sqrt(D * dt) / Q1
        sigmaDM = np.sqrt(D * dt + v**2 * dt**2) / Q2        
                        
        params = [
            {"D": 0.02, "dt": 1, "r_cs": r_c, "sigmaCD": sigmaCD, "Ns": [50] * 6},  # Confined diffusion
            # {"D": 0.02, "dt": 1, "vs": v, "sigmaDM": sigmaDM, "Ns": [50] * 6},   # Directed diffusion
            {"D": 0.02, "dt": 1, "sigma1s": sigmaND, "Ns": [50] * 6},  # Normal diffusion
        ]

        # print("Confinement radii: ", r_c)
        # print("Velocity: ", v)

        mixed_trace = Get_shifting_diff(diff_gens, params, state, lens, plot=True)
        tracks.append(mixed_trace)

        track_labels = []
        for s, l in zip(state, lens):
            label = "Confined" if s == 0 else "Normal"
            track_labels.extend([label] * l)
        labels.append(track_labels)

    return tracks, labels


def main():
    
    # diff_gens = [Gen_confined_diff_3D, Gen_directed_diff_3D]
    diff_gens = [Gen_confined_diff_3D, Gen_normal_diff_3D]

    state = [0, 1] * 6  # Alternating between Confined (0) and Directed (1) diffusion
    lens = [50] * 12    # Each segment lasts 50 frames (12 segments total)

    print("Generating mixed diffusion tracks...")
    num_tracks = 100
    tracks, labels = generate_mixed_diffusion_tracks(num_tracks, diff_gens, state, lens)

    return tracks,labels

if __name__ == "__main__":
    tr,lbl= main()
    
    print("Saving tracks and labels...")
    with open("mixed_binary_diffusion_tracks_CD_ND_3D.pkl", "wb") as f:
        pickle.dump((tr, lbl), f)
    
    print("Tracks and labels saved successfully!")
