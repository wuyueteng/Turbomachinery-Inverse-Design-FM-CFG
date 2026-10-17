# Data dictionary and provenance

All arrays are numeric, finite, physical/raw values, not pre-normalized. Geometry and labels must be paired by row.

| Geometry file | Label file | Rows | Label source |
|---|---|---:|---|
| `parameter_geometry_5200.npy` | `label_5200.npy` | 3,840 | CFD, Dataset I |
| `parameters-10000.npy` | `performance-10000.npy` | 10,000 | TeM predictions |
| `merged_geometry.npy` | `merged_label.npy` | 13,840 | Both sources, Dataset II |

The `_5200` suffix is a historical name, not the row count. Geometry has 21 columns and labels have three: **mass flow (kg/s), isentropic efficiency (fraction), total pressure ratio (dimensionless)**.

Geometry comprises seven columns for each section, ordered hub, midspan, tip:

| Within-section index (0-based) | Stored alias | Meaning / stored unit |
|---:|---|---|
| 0 | `Angle_in` | Inlet metal angle, degrees |
| 1 | `Angle_out` | Outlet metal angle, degrees |
| 2 | `Chord` | Chord length, metres |
| 3 | `THmax/CH` | Maximum relative thickness, percentage points (e.g. 12, not 0.12) |
| 4 | `THmaxP` | Maximum-thickness position, chord fraction (e.g. 0.35, not 35) |
| 5, 6 | `SWA`, `BOWA` | Section displacement parameters, metres; retain the stored coordinate ordering |

The legacy `SWA`/`BOWA` aliases are not angle units. Section bend/sweep definitions are given in the manuscript parameterization; CSV conversion does not reorder these two coordinates. The stored thickness-position fraction differs from the percentage presentation in Supplementary Table 2.

CSV exports combine 21 geometry columns and three performance columns in the same order: `dataset_I_CFD.csv`, `additional_TeM.csv`, `dataset_II_merged.csv`. Values are exported with 17 significant digits and checked for exact roundtrip.

`sample_sources.csv` maps each merged row to its CFD or TeM source row. All indices are zero-based. The merged arrays are a permutation of the two paired input sets; they are not published in simple concatenation order. That original order is preserved because it affects the seeded splits.

The generator's 80/10/10 split uses NumPy permutation seed 42 by default; the predictor uses Torch permutation seed 15. The complete reference CFD arrays provide min/max scaling, including when training on merged data. Both the normalization basis and saved split indices are documented explicitly; the unified entrypoint does not change the historical reference scale.

Data are publicly distributed under CC BY 4.0; see `../LICENSE-DATA`. Run `python scripts/check_data.py --check-results` from the repository root to verify arrays, row mappings, CSV values and archived file hashes.
