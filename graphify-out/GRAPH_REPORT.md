# Graph Report - capstone  (2026-09-19)

## Corpus Check
- 14 files · ~165,487 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 4)

## Summary
- 58 nodes · 84 edges · 12 communities (10 shown, 2 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.93)
- Token cost: 1,250 input · 980 output

## Community Hubs (Navigation)
- Image Quality Extraction Pipeline
- Presentation Figure Infrastructure
- Scientific Chart Generators
- Project Scope & Research Documentation
- Statistical Analysis Core Dependencies
- Environmental Condition Comparisons
- Hypothesis Testing & Effect Sizes
- Degradation Gallery Visualizer
- Photometric Correlation Analysis
- Descriptive Statistics Engine
- Master Presentation Dashboard
- Multi-Task ViT Architecture

## God Nodes (most connected - your core abstractions)
1. `save_dual_format()` - 7 edges
2. `main()` - 6 edges
3. `main()` - 6 edges
4. `plot_comparison()` - 5 edges
5. `perform_hypothesis_tests()` - 4 edges
6. `plot_correlations()` - 4 edges
7. `plot_extreme_samples_gallery()` - 4 edges
8. `generate_master_presentation_dashboard()` - 4 edges
9. `Phase 1 BDD100K Environmental Characterization` - 4 edges
10. `compute_group_stats()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `plot_comparison()` --shares_data_with--> `Clear vs Rain Distribution Plot`  [INFERRED]
  src/analyze_and_plot.py → results/figures/clear_vs_rain_distributions.png
- `plot_comparison()` --shares_data_with--> `Day vs Night Distribution Plot`  [INFERRED]
  src/analyze_and_plot.py → results/figures/day_vs_night_distributions.png
- `Mann-Whitney U, Kolmogorov-Smirnov & Cohen d Effect Sizes` --implements--> `perform_hypothesis_tests()`  [INFERRED]
  graphify-out/converted/autonomous_vehicle_vit_capstone_report_fbd17e5d.md → src/analyze_and_plot.py
- `plot_correlations()` --shares_data_with--> `Visual Quality Metrics Correlation Matrix`  [INFERRED]
  src/analyze_and_plot.py → results/figures/visual_correlations.png
- `plot_extreme_samples_gallery()` --shares_data_with--> `Extreme Environmental Samples Gallery`  [INFERRED]
  src/analyze_and_plot.py → results/figures/extreme_samples_gallery.png

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **BDD100K Photometric Characterization & Visualization Pipeline** — src_calculate_image_quality, src_analyze_and_plot, src_generate_presentation_illustrations, readme_photometric_metrics, docs_tech_report_phase1_characterization [INFERRED 0.95]
- **Empirical Metric Distribution & Hypothesis Testing Figures** — results_figures_clear_vs_rain_distributions_figure, results_figures_day_vs_night_distributions_figure, results_figures_visual_correlations_figure, results_figures_figure3_luminance_cdf_linechart_figure, docs_tech_report_statistical_tests [INFERRED 0.85]

## Communities (12 total, 2 thin omitted)

### Community 0 - "Image Quality Extraction Pipeline"
Cohesion: 0.18
Nodes (12): concurrent_futures, cv2, os, pandas, compute_image_metrics(), load_metadata(), main(), BDD100k Image Quality Statistics & Visual Properties Extractor Calculates: -… (+4 more)

### Community 1 - "Presentation Figure Infrastructure"
Cohesion: 0.29
Nodes (6): matplotlib_patches, matplotlib_pyplot, pathlib, Cumulative Exposure & Dynamic Range CDF Line Chart, shutil, BDD100k Presentation-Ready Scientific Illustrations Generator Produces…

### Community 2 - "Scientific Chart Generators"
Cohesion: 0.43
Nodes (7): generate_correlation_heatmap(), generate_domain_benchmark_bargraph(), generate_environmental_histograms(), generate_luminance_cdf_linechart(), main(), Saves a matplotlib figure in both SVG and high-resolution JPG formats., save_dual_format()

### Community 3 - "Project Scope & Research Documentation"
Cohesion: 0.33
Nodes (6): Capstone Review-1 Project Report & Literature Review, Phase 1 BDD100K Environmental Characterization, BDD100K Autonomous Driving Dataset, Photometric Quality Metrics (Luminance, Contrast, Blur, Exposure), Capstone Multi-Task AV Perception Project, Vision Transformer Robustness & Multi-Task Benchmark

### Community 4 - "Statistical Analysis Core Dependencies"
Cohesion: 0.33
Nodes (5): json, matplotlib_ticker, numpy, scipy, BDD100k Image Quality Statistics Analysis & Visualization Engine Generates: 1.…

### Community 5 - "Environmental Condition Comparisons"
Cohesion: 0.50
Nodes (4): Clear vs Rain Distribution Plot, Day vs Night Distribution Plot, plot_comparison(), Creates a multi-panel publication quality comparison figure (5 subplots). Each…

### Community 6 - "Hypothesis Testing & Effect Sizes"
Cohesion: 0.67
Nodes (3): Mann-Whitney U, Kolmogorov-Smirnov & Cohen d Effect Sizes, perform_hypothesis_tests(), Computes Mann-Whitney U, KS-test, and Cohen's d.

### Community 7 - "Degradation Gallery Visualizer"
Cohesion: 0.67
Nodes (3): Extreme Environmental Samples Gallery, plot_extreme_samples_gallery(), Identifies and plots 8 visual exemplar images representing quality extremes: 1.…

### Community 8 - "Photometric Correlation Analysis"
Cohesion: 0.67
Nodes (3): Visual Quality Metrics Correlation Matrix, plot_correlations(), Generates cross-metric visual correlation scatter and density maps.

### Community 9 - "Descriptive Statistics Engine"
Cohesion: 0.67
Nodes (3): compute_group_stats(), main(), Computes comprehensive statistics for each metric.

## Knowledge Gaps
- **11 isolated node(s):** `Vision Transformer Robustness & Multi-Task Benchmark`, `BDD100K Autonomous Driving Dataset`, `Photometric Quality Metrics (Luminance, Contrast, Blur, Exposure)`, `Mann-Whitney U, Kolmogorov-Smirnov & Cohen d Effect Sizes`, `Multi-Task ViT Perception System (Detection, Segmentation, Drivable Area)` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 28 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Phase 1 BDD100K Environmental Characterization` connect `Project Scope & Research Documentation` to `Image Quality Extraction Pipeline`?**
  _High betweenness centrality (0.164) - this node is a cross-community bridge._
- **Why does `plot_comparison()` connect `Environmental Condition Comparisons` to `Descriptive Statistics Engine`, `Statistical Analysis Core Dependencies`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `perform_hypothesis_tests()` connect `Hypothesis Testing & Effect Sizes` to `Descriptive Statistics Engine`, `Statistical Analysis Core Dependencies`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `plot_comparison()` (e.g. with `Clear vs Rain Distribution Plot` and `Day vs Night Distribution Plot`) actually correct?**
  _`plot_comparison()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Vision Transformer Robustness & Multi-Task Benchmark`, `BDD100K Autonomous Driving Dataset`, `Photometric Quality Metrics (Luminance, Contrast, Blur, Exposure)` to the rest of the system?**
  _11 weakly-connected nodes found - possible documentation gaps or missing edges._