# Research input schema

Arrays contain raw physical geometry and performance values; loaders perform normalization.

| Filename | Shape in the supplied research workspace | Meaning |
|---|---|---|
| `parameter_geometry_5200.npy` | `(3840, 21)` | CFD reference geometry |
| `label_5200.npy` | `(3840, 3)` | Corresponding CFD performance |
| `merged_geometry.npy` | `(13840, 21)` | Augmented geometry dataset |
| `merged_label.npy` | `(13840, 3)` | Corresponding performance labels |

The `_5200` suffix is a historical filename, not the current row count. The generative data loader uses the merged arrays by default; all four files are needed for that workflow.

Performance columns are **mass flow (kg/s), isentropic efficiency, total pressure ratio**, in that order. Do not interchange the second and third columns.

Geometry columns are seven parameters for each of three sections, in the order defined by `shared.data_utils.GEOM_NAMES`: `Angle_in`, `Angle_out`, `Chord`, `THmax/CH`, `THmaxP`, `SWA`, `BOWA`, repeated for sections 1, 2 and 3. Retain the source dataset's units and section ordering when supplying compatible inputs.

Shared normalization uses the minimum and maximum of each column of the full CFD reference arrays. The same reference scale is used for augmented data and the surrogate. The predictor dataset likewise computes min/max from its supplied arrays before partitioning. These implementation choices are retained in this release.

Data remain available from the corresponding authors upon reasonable request; no raw arrays are included here.
