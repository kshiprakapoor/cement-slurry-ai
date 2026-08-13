"""
Generate a synthetic cement slurry thickening-time dataset.

IMPORTANT:
- This dataset is synthetic and intended only for portfolio/demo purposes.
- The relationships are directionally plausible, not calibrated to field or laboratory data.
- Do not use model outputs for operational cementing decisions.
"""

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_WELLS = 1000
OUTPUT_FILE = Path(__file__).resolve().parent / "slurry_performance_data.csv"


def generate_dataset(n_wells: int = N_WELLS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Depth: operational range requested by the project.
    depth_ft = rng.uniform(5000, 15000, n_wells)

    # BHCT is generated with a loose positive relationship to depth,
    # plus well-to-well variability, then constrained to the requested range.
    bhct_f = 75 + 0.013 * depth_ft + rng.normal(0, 18, n_wells)
    bhct_f = np.clip(bhct_f, 100, 300)

    # Retarder concentration, % BWOC.
    retarder_pct_bwoc = rng.uniform(0.1, 1.5, n_wells)

    # Slurry density. Centered near 16.2 ppg, bounded to requested range.
    slurry_density_ppg = rng.normal(16.2, 0.55, n_wells)
    slurry_density_ppg = np.clip(slurry_density_ppg, 15.0, 17.5)

    # Synthetic thickening-time relationship:
    # - Higher BHCT generally accelerates hydration -> shorter thickening time.
    # - Higher retarder concentration generally extends thickening time.
    # - Depth is included as a mild proxy for downhole severity/pressure context.
    # - Density has a modest synthetic effect.
    # - Temperature x retarder interaction allows retarder dosage to partially
    #   compensate for higher-temperature conditions.
    # - Random noise represents unmodeled formulation/measurement variability.
    thickening_time_min = (
        320
        - 1.15 * (bhct_f - 180)
        + 190 * (retarder_pct_bwoc - 0.60)
        + 45 * (retarder_pct_bwoc - 0.60) ** 2
        - 0.003 * (depth_ft - 10000)
        - 16 * (slurry_density_ppg - 16.2)
        + 0.38 * (bhct_f - 180) * (retarder_pct_bwoc - 0.70)
        + rng.normal(0, 18, n_wells)
    )

    # Bound the synthetic target to a plausible engineering-demo range.
    thickening_time_min = np.clip(thickening_time_min, 45, 720)

    df = pd.DataFrame(
        {
            "Well_ID": [f"WELL_{i:04d}" for i in range(1, n_wells + 1)],
            "BHCT_F": np.round(bhct_f, 1),
            "Depth_ft": np.round(depth_ft, 0).astype(int),
            "Retarder_pct_BWOC": np.round(retarder_pct_bwoc, 3),
            "Slurry_Density_ppg": np.round(slurry_density_ppg, 2),
            "Thickening_Time_min": np.round(thickening_time_min, 1),
        }
    )

    return df


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(data):,} synthetic wells to: {OUTPUT_FILE}")
    print("\nDataset summary:")
    print(data.select_dtypes(include="number").describe().round(2))
