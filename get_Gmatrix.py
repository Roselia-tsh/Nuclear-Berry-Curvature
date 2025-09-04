#!/usr/bin/env python3
import os
import numpy as np
import re

Area = 4.9e-5  # Å²
h = 6.5821251e-16
directions = ['x', 'y', 'z']


def count_atoms_in_poscar(poscar_path):
    with open(poscar_path, 'r') as f:
        lines = f.readlines()
        num_line = None
        for idx, line in enumerate(lines):
            if line.strip().split()[0].isdigit():
                num_line = idx
                break
        if num_line is None:
            raise RuntimeError(f"Cannot parse atom numbers from {poscar_path}")
        numbers = list(map(int, lines[num_line].split()))
        return sum(numbers)


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
        print(f"[Warning] Missing file: {folder}/{key}/WSWQ")
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


def compute_G_matrix(i, j, Nk, Nb):
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

            phi = 0.0
            for k in range(Nk):
                M_total = M01[k] @ M12[k] @ M20[k]
                phi_k = -np.angle(np.linalg.det(M_total))
                phi += phi_k

            phi_avg = phi / Nk
            G[a, b] = 2 * h * phi_avg / Area

    return G


def main():
    n_atoms = count_atoms_in_poscar('SPOSCAR')
    Oi_dir = "matrix_0-1"
    Oi_key = "0-1x"
    M01_sample = get_overlap_matrix(Oi_dir, Oi_key)
    if M01_sample is not None:
        Nk = M01_sample.shape[0]
        Nb = M01_sample.shape[1]
        print(f"Detected Nk = {Nk}, Nb = {Nb}")
    else:
        print("Error: Cannot read WSWQ for 1x direction.")
        return

    G_dict = {}

    for i in range(1, n_atoms + 1):
        for j in range(i, n_atoms + 1):
            G_ij = compute_G_matrix(i, j, Nk, Nb)
            G_dict[(i, j)] = G_ij
            print(f"pair {i}-{j} : done")

    with open("G_CONSTANT", "w") as f:
        f.write(f"{n_atoms} {n_atoms}\n")
        for i in range(1, n_atoms + 1):
            for j in range(1, n_atoms + 1):
                if (i, j) in G_dict:
                    G = G_dict[(i, j)]
                elif (j, i) in G_dict:
                    G = -G_dict[(j, i)].T
                else:
                    G = np.zeros((3, 3))

                f.write(f"{i} {j}\n")
                for row in G:
                    f.write(" ".join(f"{val: .6e}" for val in row) + "\n")

    print("\nAll G matrices saved to G_CONSTANT.")


if __name__ == "__main__":
    main()

