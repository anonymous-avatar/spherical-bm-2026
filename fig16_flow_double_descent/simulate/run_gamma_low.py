# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.6",
#     "numpy>=2.0",
# ]
# ///
"""Low-γ extension: γ ∈ {0.01, 0.02, 0.05, 0.1}.

Re-uses run_gamma.py's pipeline; just overrides the GAMMAS list.
"""
import argparse
import sys

import numpy as np

import run_gamma                                      # noqa: E402

# Override gamma grid in the imported module
run_gamma.GAMMAS = np.array([0.01, 0.02, 0.05, 0.1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma-idx", type=int, required=True)
    p.add_argument("--outdir", type=str, default="results_low")
    args = p.parse_args()
    sys.argv = ["run_gamma_low.py", "--gamma-idx", str(args.gamma_idx),
                "--outdir", args.outdir]
    run_gamma.main()


if __name__ == "__main__":
    main()
