# CFD-SOM Adaptive Meshing

A reproducible computational research project exploring **Self-Organizing Maps (SOMs) for adaptive mesh refinement in CFD**, combining OpenFOAM simulations, Python scientific computing, automated numerical experiments, sensitivity analysis, and quantitative validation against a fine-grid numerical reference.

## Research Objective

The central question is:

> Can a Self-Organizing Map trained on CFD solution features identify flow-field structures that are useful for adaptive mesh refinement without using the fine-grid reference during cell selection?

The purpose is not to demonstrate that SOM-based refinement is universally superior to conventional gradient-based refinement. The project investigates where unsupervised flow-state classification provides useful information, where it does not, and how it may complement conventional CFD indicators.

## CFD Benchmark

The study uses a lid-driven cavity with:

- Domain: 0.1 m x 0.1 m
- Lid velocity: 1 m/s
- Kinematic viscosity: 0.001 m2/s
- Reynolds number: Re = 100
- Transient incompressible CFD
- OpenFOAM v2606

Four uniform meshes are investigated:

- M20
- M40
- M80
- M160

M160 is used as a **fine-grid numerical reference**, not as an exact analytical solution.

## SOM Methodology

The nominal SOM is trained from the M40 CFD solution using standardized features:

```text
[Ux*, Uy*, |U|*, |grad(U)|*]
```

Spatial coordinates are excluded from the SOM input.

Nominal configuration:

- Architecture: 4 x 4
- Neurons: 16
- Training iterations: 20,000
- Random seed: 42
- Initial learning rate: 0.5
- Final learning rate: 0.05
- Initial neighborhood width: 2.0
- Final neighborhood width: 0.5

Reproduced nominal results:

- Mean quantization error: 0.497592782
- Maximum quantization error: 7.514196525
- Exact BMU reproducibility: 100%

## Refinement Strategies

### Gradient

A velocity-gradient-based indicator provides the principal conventional baseline.

### SOM Quantization Error

SOM quantization error (QE) was investigated as a direct refinement indicator.

The experiments show that QE should **not** be interpreted as a direct CFD discretization-error estimator. Gradient-based selection generally outperformed direct SOM-QE selection over the tested refinement budgets.

### Hybrid Indicator

A hybrid formulation combining normalized gradient information with SOM-classified-state transition information was investigated.

It was competitive in some cases but did not demonstrate systematic superiority over the gradient baseline.

### SOM-Interface Strategy

A subsequent strategy identifies interfaces between neighboring cells assigned to different SOM Best Matching Units (BMUs).

Priority interface cells are selected using local SOM transition information together with velocity-gradient and velocity-jump information. Selected seeds are expanded through mesh-neighbor connectivity to generate refinement regions.

The M160 reference is **not used during cell selection**.

## Cell-Matched Adaptive Comparison

For the tested cavity configuration:

| Strategy | Final cells | Mean error | RMS error |
|---|---:|---:|---:|
| SOM-interface 1N | 1834 | 0.001570 | 0.003929 |
| Gradient matched 1N | 1834 | 0.001824 | 0.003986 |
| SOM-interface 2N | 1942 | 0.001359 | 0.003611 |
| Gradient matched 2N | 1942 | 0.001672 | 0.003761 |

For these specific matched comparisons, SOM-interface refinement produced lower mean and RMS velocity errors relative to M160. Gradient refinement produced slightly lower maximum errors.

These results are **case-specific and do not establish general superiority of SOM-based adaptive meshing**.

## SOM Sensitivity Study

The sensitivity analysis covers:

- 3 x 3 SOM
- 4 x 4 SOM
- 5 x 5 SOM
- 6 x 6 SOM
- Five random seeds per architecture
- 20 SOM configurations in total

The analysis examines:

- Quantization error
- Neuron occupancy
- Partition stability
- Interface-cell fraction
- Priority-cell overlap
- Spatial proximity of selected refinement seeds

Increasing SOM resolution generally decreases quantization error while increasing the fraction of cells classified as interfaces.

Therefore, lower SOM quantization error alone is not evidence of a better adaptive-mesh indicator.

The original 4 x 4 SOM remains the nominal configuration. The sensitivity study characterizes robustness rather than retrospectively optimizing the model against M160.

## Exact SOM Reproducibility

The original 4 x 4 SOM can be reproduced with:

```bash
python3 python/test_som_reproducibility.py
```

A successful reference test produces:

```text
Exact BMU equality     : True
BMU match fraction     : 1.000000000
Max weight difference  : 0.000000000000e+00

PASS: original SOM reproduced exactly.
```

This test verifies deterministic reproduction of the nominal SOM configuration, including BMU assignments and trained weights.

## Repository Structure

```text
CFD-SOM-Adaptive-Meshing/
|-- cases/       OpenFOAM benchmark and adaptive cases
|-- data/        Processed CFD and SOM datasets
|-- framework/   Experiment-management and validation utilities
|-- python/      SOM, CFD analysis, refinement and plotting scripts
|-- report/      Technical report, LaTeX source and figures
|-- results/     Numerical comparisons and sensitivity results
|-- README.md
`-- requirements.txt
```

Large transient OpenFOAM solution histories, visualization exports, caches, logs, and temporary files are intentionally excluded from version control.

The included OpenFOAM cases retain input and configuration files together with relevant mesh definitions needed to document the tested configurations without storing unnecessary generated solution histories.

## Python Environment

Reference environment:

```text
Python 3.12.3
numpy 1.26.4
pandas 2.1.4
matplotlib 3.6.3
scipy 1.11.4
scikit-learn 1.4.1.post1
```

Install the Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## OpenFOAM Environment

The reference CFD calculations were performed using **OpenFOAM v2606**.

Users reproducing the CFD calculations should verify solver and dictionary compatibility if using a different OpenFOAM release.

## Important Interpretation and Limitations

The SOM learns patterns from a **discrete CFD solution**. It should not be interpreted as independently discovering exact physical flow regimes.

Important limitations include:

- SOM quantization error measures representation novelty in the standardized feature space, not CFD discretization error.
- BMU interfaces depend on SOM architecture, initialization, training procedure, and feature selection.
- Velocity-gradient magnitude is itself one of the SOM input features, so SOM-derived structures are not completely independent of conventional gradient information.
- Correlation between an indicator and reference error does not by itself establish refinement utility.
- Refinement utility must ultimately be evaluated after solving the governing equations on the refined mesh.
- M160 is a fine-grid numerical reference rather than an exact solution.
- The present investigation focuses on a single lid-driven-cavity benchmark.
- The adaptive study is primarily one-shot rather than a fully iterative solve-estimate-refine cycle.
- Broader validation across Reynolds numbers, geometries, flow conditions, and transport problems is required before general conclusions can be drawn.

The current results therefore support continued investigation rather than a universal claim about SOM-based adaptive meshing.

## Future Work

Potential extensions include:

- Additional CFD benchmark problems
- Different Reynolds numbers
- Iterative adaptive-refinement cycles
- Alternative physically motivated feature sets
- Direction-sensitive flow features
- Thermal and scalar-transport problems
- Evaluation of computational cost versus accuracy
- Additional comparisons with conventional refinement indicators
- Assessment of SOM scalability for larger CFD meshes
- Development of reusable automated CFD/ML refinement workflows

## Technical Report

The complete technical report is included in `report/main.pdf`.

The report documents the numerical methodology, mesh study, SOM formulation, adaptive-refinement experiments, sensitivity analysis, results, limitations, and discussion.

LaTeX source files and report figures are also included in the `report/` directory.

## Reproducibility Philosophy

This repository preserves both successful and unsuccessful research directions. In particular, the direct SOM-QE refinement approach is retained even though it did not outperform the gradient baseline in the tested refinement-budget study.

Negative results are important for understanding what the SOM representation does and does not measure, and they motivated the subsequent SOM-interface investigation.

The project emphasizes reproducibility, numerical verification, controlled comparisons, explicit limitations, and separation between observed results and broader interpretation.

## Scope

This is an **independent engineering research project** intended for reproducible computational research and technical demonstration.

Only independently developed material, general/public scientific methods, and project-specific benchmark data are included.

The repository is not intended to contain proprietary employer information, confidential engineering data, or project-specific information from professional employment.

## Author

**José David Jiménez Díaz**

Independent Engineering Research
Mississauga, Ontario, Canada

Email: jd.jimenez.eng@gmail.com

LinkedIn: https://www.linkedin.com/in/josedavidjimenezdiazeng/

GitHub: https://github.com/JoseJimenezD/CFD-SOM-Adaptive-Meshing

## Status

**Research prototype**

The current results justify further investigation of SOM-assisted mesh-selection strategies, but broader CFD validation is required before general conclusions about performance or applicability can be made.

