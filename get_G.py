#!/usr/bin/env python3
import os
import numpy as np
import argparse
import re

Area = 1e-4  # Å²

h = 6.5821251e-16
directions = ['x', 'y', 'z']

def parse_args():
    parser = argparse.ArgumentParser(description="Compute G_{iα,jβ} matrix using Berry curvature formula.")
    parser.add_argument("-p", nargs=2, type=int, required=True, metavar=('i', 'j'),
                        help="Atom indices i and j (positive integers).")
    return parser.parse_args()

def load_wswq(path):
    if not os.path.exists(path):
        return None

    current_k = -1
    data_dict = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("spin="):
                match = re.search(r"kpoint=\s*(\d+)", line)
                if match:
                    current_k = int(match.group(1)) - 1
                    data_dict[current_k] = {}
            elif line.startswith("i="):
                match = re.match(r"i=\s*(\d+),\s*j=\s*(\d+)\s*:\s*([-0-9eE.]+)\s+([-0-9eE.]+)", line)
                if match:
                    i = int(match.group(1)) - 1
                    j = int(match.group(2)) - 1
                    re_val = float(match.group(3))
                    im_val = float(match.group(4))
                    data_dict[current_k][(i, j)] = re_val + 1j * im_val

    if not data_dict:
        return None

    nk = max(data_dict.keys()) + 1
    nb = max(max(i, j) for d in data_dict.values() for (i, j) in d.keys()) + 1
    data = np.zeros((nk, nb, nb), dtype=complex)

    for k in data_dict:
        for (i, j), val in data_dict[k].items():
            data[k, i, j] = val

    return data

def hermitian_conjugate(matrix):
    return matrix.conj().transpose(0, 2, 1)

def get_overlap_matrix(folder, key):
    path = os.path.join(folder, key, "WSWQ")
    if os.path.exists(path):
        return load_wswq(path)
    else:
        print(f"[Warning] Missing file: please provide forward overlap '{key}' in '{folder}'")
        return None

def get_middle_matrix(i, j, alpha, beta, M01):
    folder = f"matrix_{i}-{j}"
    key_fwd = f"{i}{alpha}-{j}{beta}"
    key_rev = f"{j}{beta}-{i}{alpha}"

    if i == j:
        if alpha == beta:
            return np.eye(M01.shape[1])[None, :, :].repeat(M01.shape[0], axis=0)
        elif alpha < beta:
            return get_overlap_matrix(folder, key_fwd)
        else:
            mat = get_overlap_matrix(folder, key_rev)
            return hermitian_conjugate(mat) if mat is not None else None
    else:
        return get_overlap_matrix(folder, key_fwd)

def compute_G_matrix(i, j):
    G = np.zeros((3, 3))
    for a, alpha in enumerate(directions):
        for b, beta in enumerate(directions):
            if i == j:
                if a > b:
                    G[a, b] = -G[b, a]
                    continue
                elif a == b:
                    G[a, b] = 0.0
                    continue

            Oi_dir = f"matrix_0-{i}"
            Oj_dir = f"matrix_0-{j}"
            Oi_key = f"0-{i}{alpha}"
            Oj_key = f"0-{j}{beta}"

            M01 = get_overlap_matrix(Oi_dir, Oi_key)
            M20 = get_overlap_matrix(Oj_dir, Oj_key)
            if M01 is None or M20 is None:
                continue
            M20 = hermitian_conjugate(M20)

            M12 = get_middle_matrix(i, j, alpha, beta, M01)
            if M12 is None:
                continue

            Nk = M01.shape[0]
            phi = 0.0
            for k in range(Nk):
                M_total = M01[k] @ M12[k] @ M20[k]
                phi_k = -np.angle(np.linalg.det(M_total))
                phi += phi_k

            phi_avg = phi / Nk
            G[a, b] = 2 * h * phi_avg / Area

    return G

def main():
    args = parse_args()
    i, j = args.p
    assert i > 0 and j > 0, "Atom indices must be positive integers"

    if i > j:
        i, j = j, i
    
    Oi_dir = f"matrix_0-{i}"
    Oi_key = f"0-{i}x"
    M01_sample = get_overlap_matrix(Oi_dir, Oi_key)
    if M01_sample is not None:
        Nk = M01_sample.shape[0]
        Nb = M01_sample.shape[1]
        print(f"Detected Nk = {Nk}, Nb = {Nb}")

    G_ij = compute_G_matrix(i, j)
    print(f"Computed G matrix for atom pair ({i}, {j})")
    print("G_{iα, jβ} matrix [eV*s/Å²]:")
    for row in G_ij:
        print("  ".join(f"{val: .6e}" for val in row))

    if i != j:
        G_ji = -G_ij.T
        print(f"\nInferred G matrix for atom pair ({j}, {i}) using antisymmetry")
        print("G_{jβ, iα} matrix [eV*s/Å²]:")
        for row in G_ji:
            print("  ".join(f"{val: .6e}" for val in row))

main()

