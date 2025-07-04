#!/usr/bin/env python3
import argparse
import numpy as np
import sys

def parse_wswq(filename, target_kpoint, target_i=None, target_j=None):
    """
    Parse the WSWQ file and compute the normalization sum for either:
    - fixed i: sum over all j of |<psi_j | psi_i>|^2
    - fixed j: sum over all i of |<psi_j | psi_i>|^2
    at a given k-point.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    current_kpoint = None
    recording = False
    mod_squared_sum = 0.0

    for line in lines:
        line = line.strip()

        # Detect and select the target k-point block
        if line.startswith("spin="):
            parts = line.split(',')
            for part in parts:
                if 'kpoint=' in part:
                    current_kpoint = int(part.split('=')[1])
                    recording = (current_kpoint == target_kpoint)
                    break
            continue

        # Process overlap entries only within the target k-point block
        if recording and line.startswith("i="):
            parts = line.replace(":", "").split()
            i = int(parts[1].strip(','))
            j = int(parts[3].strip(','))
            real = float(parts[4])
            imag = float(parts[5])
            value = complex(real, imag)

            # Case 1: fixed i, sum over all j
            if target_i is not None and i == target_i:
                mod_squared_sum += abs(value)**2

            # Case 2: fixed j, sum over all i
            if target_j is not None and j == target_j:
                mod_squared_sum += abs(value)**2

        # If a new spin block starts, stop processing the current block
        elif recording and line.startswith("spin="):
            break

    return mod_squared_sum

def main():
    parser = argparse.ArgumentParser(description="Check normalization of overlap matrix rows or columns.")
    parser.add_argument("-k", type=int, required=True, help="K-point index (starting from 1)")
    parser.add_argument("-i", type=int, help="Band index i (fix i, sum over j)")
    parser.add_argument("-j", type=int, help="Band index j (fix j, sum over i)")
    parser.add_argument("-f", "--file", type=str, default="WSWQ", help="WSWQ filename (default: WSWQ)")
    args = parser.parse_args()

    # Validate input: must provide exactly one of -i or -j
    if (args.i is None and args.j is None) or (args.i is not None and args.j is not None):
        print("Error: You must provide exactly one of -i or -j (not both, and not neither).", file=sys.stderr)
        sys.exit(1)

    # Compute the normalization sum
    total = parse_wswq(args.file, args.k, target_i=args.i, target_j=args.j)

    # Output result
    if args.i is not None:
        print(f"Sum of |<psi_j | psi_{args.i}>|^2 over j at k-point {args.k} = {total:.6f}")
    elif args.j is not None:
        print(f"Sum of |<psi_{args.j} | psi_i>|^2 over i at k-point {args.k} = {total:.6f}")

if __name__ == "__main__":
    main()

