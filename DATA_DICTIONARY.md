# Cement Slurry Thickening-Time Dataset — Data Dictionary

> **Portfolio/demo dataset only.** All records and target values are synthetic.  
> The generated relationships are directionally plausible but are **not** calibrated to field data,
> laboratory schedules, proprietary cement systems, API/ISO test protocols, or Halliburton models.
> The dataset and resulting ML model must not be used for operational cementing decisions.

| Feature | Data Type | Unit | Operational / Generated Bounds | Description |
|---|---|---:|---:|---|
| `Well_ID` | string | — | `WELL_0001`–`WELL_1000` | Synthetic unique identifier for each well/sample record. It is excluded from model training. |
| `BHCT_F` | float | °F | 100–300 °F | Bottomhole Circulating Temperature (BHCT). Represents the approximate circulating temperature relevant to slurry placement conditions. In the synthetic generator, BHCT is loosely correlated with depth. |
| `Depth_ft` | integer | ft | 5,000–15,000 ft | Synthetic well depth. Used as a contextual downhole variable and mild proxy for increasing downhole severity/pressure. |
| `Retarder_pct_BWOC` | float | % BWOC | 0.1–1.5% | Retarder concentration expressed as percent by weight of cement (BWOC). Higher retarder dosage is modeled to generally increase slurry thickening time. |
| `Slurry_Density_ppg` | float | lb/gal (ppg) | 15.0–17.5 ppg | Cement slurry density. Included as a formulation/property feature with a modest synthetic influence on thickening time. |
| `Thickening_Time_min` | float | min | 45–720 min | Synthetic target variable: predicted time until the slurry reaches a defined thickening endpoint in a hypothetical lab-style test. It is generated from nonlinear feature effects, interactions, and random noise. |

## Synthetic Relationship Assumptions

The generator intentionally encodes several **directional** relationships so the dataset is more meaningful than independent random numbers:

- Higher **BHCT** generally reduces thickening time.
- Higher **retarder concentration** generally increases thickening time.
- **Depth** has a comparatively mild modeled effect.
- **Slurry density** contributes a smaller formulation effect.
- A **BHCT × retarder interaction** allows higher retarder dosage to partially offset high-temperature acceleration.
- Random noise represents formulation differences, measurement variation, and omitted variables.

These assumptions are useful for demonstrating a data-science workflow, but real cement slurry behavior depends on many additional variables such as cement chemistry, additive package, pressure, mixing history, fluid loss, rheology, water chemistry, test schedule, and operational design constraints.
