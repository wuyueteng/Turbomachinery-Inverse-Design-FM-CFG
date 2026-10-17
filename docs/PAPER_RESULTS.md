# Paper figure and numerical-result map

These identifiers refer to the final manuscript and Supplementary Information. Select them with `python scripts/reproduce_figures.py --figure <ID>`, or use `--all`. Replots preserve the numerical inputs/definitions and use a portable presentation; they are not a pixel-identical export of the typeset manuscript. Results use CC BY 4.0.

| Paper item | CLI ID | Sources below `paper_results/` | Interpretation |
|---|---|---|---|
| Fig. 2 | `2` | `surrogate/` MC arrays and robustness CSV | MC-mean regression, calibration and first 100 archived training-seed records |
| Fig. 3; Table 1 | `3` | `benchmark/`, `models/*/inference/`, `cost/` | Six-model accuracy/diversity/novelty and measured computation costs |
| Fig. 6 | `6` | `trajectory/trajectory_data.npz` | Archived cFM-CFG/cDDPM paths, common PCA coordinates and FM velocity |
| Fig. 9d–f | `9def` | `models/fm/`, `models/fm_cfg/`, `ablation/cfg_ablation.json` | No-CFG versus CFG; k=5 diversity/novelty recomputed from stored geometries |
| Fig. 10 | `10` | `ablation/sweep_*.json` | Dropout, three-seed guidance sweep and Euler-step sensitivity |
| Fig. 11f–h | `11fgh` | Repository `data/` | CFD versus additional TeM-labelled performance coverage |
| Supplementary Fig. 1 | `S1` | `models/*/inference/` | Archived target-versus-prediction scatter across six models |
| Supplementary Fig. 2; Table 6 | `S2` | `models/*/inference/`, `benchmark/distribution_similarity.*` | Generated geometry distributions and distance statistics |
| Supplementary Fig. 6 | `S6` | `models/fm_cfg/hpo_history_fm_cfg.json` | History of the minimum downstream-MSE trial |
| Supplementary Fig. 7 | `S7` | `benchmark/hpo_importance.json` | Archived model-specific hyperparameter importance |

`benchmark/table1_display_values.json` contains the published rounding for comparison with recomputed values. `python scripts/verify_paper_results.py` checks Table 1, the archived benchmark metrics, distribution distances and Fig. 2 summary statistics. `source_manifest.json` records hashes of the included source artifacts and identifies scoped transformations, such as retaining only the paper's models in a multi-model JSON file.

The robustness experiment keeps split seed 15 fixed and varies training seeds. HPO histories retain original completed/pruned trial records. Reported computational timings are measurements on the archived hardware/protocol, not predictions for the reader's machine. Training/inference precision and normalization units remain in the cost source tables.

No raw network weights are required for these plots. New model predictions require training and should remain under `runs/` or `outputs/`. Mixed figures omit CFD-only panels (including Fig. 9a–c and Fig. 11a–e). CFD-flow visualizations, solver validation and geometry reconstruction are outside this network-result replotting package.

The source NumPy files, CSV values and JSON metrics remain the numerical reference. The shared k-nearest-neighbour definitions use k=5; they differ from the legacy evaluator's auxiliary pairwise-diversity statistic. Both are labelled in their respective source records.
