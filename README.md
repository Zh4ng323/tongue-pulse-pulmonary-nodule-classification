# Distinguishing Lung Cancer from Benign Pulmonary Nodules Using Features Derived from Tongue Images and Pulse Waveforms

A modular machine learning framework for classifying pulmonary nodules as benign or malignant using tongue and pulse features from Traditional Chinese Medicine (TCM). This repository accompanies our research paper on multi-modal feature integration for lung cancer screening.

> **Data availability**: The repository includes a de-identified sample dataset (`data/sample_data.csv`) to demonstrate the complete analysis pipeline. Due to privacy and data governance constraints, the sample dataset does not reproduce the specific results reported in the paper. The original de-identified dataset may be obtained from the corresponding author upon reasonable request and subject to ethical approval.

## Features

- **Pearson correlation analysis**: Tongue-tongue, pulse-pulse, and cross-modal correlation heatmaps with FDR correction
- **Canonical Correlation Analysis (CCA)**: Cross-modal association between tongue and pulse features
- **Integrated modeling (10-fold CV)**: LASSO feature selection + 5 ML models (LR, RF, SVM, ANN, XGBoost)
- **SHAP interpretability**: Beeswarm plots, decision plots, force plots, grouped feature importance
- **Correlation network visualization**: Cross-modal and differential network graphs
- **Publication-ready visualizations** (300 DPI)

## Quick Start

### Installation

```bash
git clone https://github.com/Zh4ng323/tongue-pulse-pulmonary-nodule-classification.git
cd tongue-pulse-pulmonary-nodule-classification
pip install -r requirements.txt
```

### Run Analysis

```bash
python main_v2.py
```

The interactive menu provides:
1. Pearson correlation analysis (heatmaps, cross-modal clustering)
2. Canonical correlation analysis (CCA)
3. Integrated modeling (10-fold cross-validation)
4. One-click correlation + CCA analysis
5. SHAP feature analysis plots

## Project Structure

```
├── main_v2.py                              # Main entry point
├── config.py                               # Global configuration
├── encoding_fix.py                         # Windows encoding fix
├── plot_forest_plots.py                    # Forest plot generation
├── requirements.txt                        # Python dependencies
├── data/
│   └── sample_data.csv                     # De-identified sample data (80 samples)
└── modules/
    ├── data_loader.py                      # Data loading and quality checks
    ├── simple_correlation.py               # Pearson correlation analysis
    ├── simple_correlation_improved_fusion.py  # Enhanced heatmap functions
    ├── correlation_network_plots.py        # Network graph visualization
    ├── cca_analysis.py                     # Canonical correlation analysis
    ├── integrated_modeling_v2.py           # ML modeling pipeline
    └── interpretability_plots_v2.py        # SHAP interpretability plots
```

## Sample Data

- **File**: `data/sample_data.csv`
- **Samples**: 80 de-identified cases (40 benign, 40 malignant)
- **Features**: 49 quantitative features (34 tongue + 15 pulse)
- **Target**: Binary classification (0 = Benign, 1 = Malignant)

### Feature Categories

**Tongue features (34)**: Per-all, Per-part, TB-CON, TC-CON, TB-ASM, TC-ASM, TB-ENT, TC-ENT, TB-MEAN, TC-MEAN, TB-B, TB-R, TB-G, TC-R, TC-G, TC-B, TB-H, TB-I, TB-S, TC-H, TC-I, TC-S, TB-L, TB-a, TB-b, TC-L, TC-a, TC-b, TB-Y, TB-Cr, TB-Cb, TC-Y, TC-Cr, TC-Cb

**Pulse features (15)**: h1, h3, h4, h5, t, t1, t4, t5, h3/h1, h1/t1, h4/h1, t1/t, t4/t5, w1/t, w2/t

## Using Your Own Data

To use your own dataset, modify the path in `config.py`:

```python
RAW_DATA_PATH = "path/to/your/data.csv"  # or .xlsx
```

Your data file should contain:
- A `Group` column (0 = Benign, 1 = Malignant)
- Feature columns matching the names in `config.py` (`TONGUE_FEATURES` and `PULSE_FEATURES`)

## Methodology

- **LASSO feature selection**: Performed within each CV fold on training data to prevent data leakage
- **Hierarchical LASSO for fusion mode**: Tongue and pulse features are independently selected, then combined
- **Dual-threshold system**: Overview threshold (|r| >= 0.15) for heatmaps, backbone threshold (|r| >= 0.20) for network graphs
- **FDR/Bonferroni correction**: Applied to correlation p-values
- **Bootstrap confidence intervals**: For model performance metrics
- **OOF (out-of-fold) SHAP values**: Ensures unbiased feature importance estimation

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@unpublished{zhang2026tonguepulse,
  title = {Distinguishing Lung Cancer from Benign Pulmonary Nodules Using Features Derived from Tongue Images and Pulse Waveforms: An Interpretable Machine Learning Study},
  author = {Shi, Yulin and Zhang, Guohao and Zhang, Shuyi and Dong, Changsheng and Xu, Ling and Zhang, Hongkai and Chen, Wenlian and Xu, Jiatuo},
  year = {2026},
  note = {Manuscript submitted for publication. \url{https://github.com/Zh4ng323/tongue-pulse-pulmonary-nodule-classification}}
}
```

## Contact

For data access requests, please contact the corresponding author.
