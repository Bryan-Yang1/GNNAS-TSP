# Results Analysis

This folder contains notebooks used to summarize experiment outputs and compare selectors against SBS/VBS baselines.

- `summarize_general_selectors.ipynb`: builds summary tables for selector, fixed-solver, SBS, and VBS performance.
- `analyze_best_selector_vbs_gap_wilcoxon.ipynb`: analyzes best-selector VBS gaps and Wilcoxon signed-rank tests against SBS.

The notebooks expect experiment result folders, such as `10s_v2/`, `60s_v2/`, or similar generated output directories, to be available relative to the repository root or adjusted in the notebook configuration cells.
