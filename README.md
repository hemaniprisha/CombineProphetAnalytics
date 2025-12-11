# Combine Prophet Analytics

## Overview
This repository analyzes the predictive value of NFL Combine metrics for wide receiver performance and provides an end-to-end framework for cleaning data, running statistical analysis, generating visualizations, training a predictive model, and launching an interactive application. The goal is to determine whether raw athletic testing meaningfully predicts rookie-year on-field production and to illustrate the limitations of combine-only evaluation.

The project includes:
- A full data processing and modeling pipeline
- Exploratory analysis (correlations, scatter plots, distribution checks)
- A predictive model with evaluation metrics and feature importance
- Saved visualizations
- A Streamlit app for interactive predictions
- A unified `run.py` CLI to execute the analysis and/or app

---

## Repository Structure
```
.
├── run.py                     # Unified entry point: analysis, app, or both
├── combine_analysis.py        # End-to-end data cleaning, modeling, plots, exports
├── app.py                     # Streamlit prediction interface
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── actual_vs_predicted.png    # Model vs actual performance comparison
├── correlation_heatmap.png    # Combine metric correlation matrix
├── feature_importance.png     # Feature importance from predictive model
└── scatter_plots.png          # Core exploratory figures
```

---

## How to Run

Run everything (analysis + app):
```
python run.py --all
```

Run only the analytical pipeline:
```
python run.py --analysis
```

Run only the Streamlit app:
```
python run.py --app
```

---

## Analysis Summary (Short)

### Data
- Combine WR records: ~444  
- Rookie WR season records: ~654  
- Final modeling sample: 92 WRs with complete data  

### Key Insights
- Correlations between combine tests and rookie performance are weak (top correlation ≈ 0.20).
- Combine metrics explain roughly 0–4 percent of performance variance.
- Predictive model (Gradient Boosting) performs below baseline, with negative test R².
- Typical error is over 50 percent of a rookie WR’s average production.
- Athletic tests like the 40-yard dash, shuttle, and vertical jump show minimal actionable predictive value.

### Interpretation
Combine performance alone does not meaningfully predict rookie-year on-field production for wide receivers. The analysis highlights the limitations of pure athletic testing and suggests that evaluation systems require contextual, game-derived metrics (tracking data, route performance, separation, etc.) for reliable projection.

All outputs (plots and model artifacts) are saved automatically:
- correlation_heatmap.png  
- scatter_plots.png  
- feature_importance.png  
- actual_vs_predicted.png  
- combine_analysis_export.pkl  

---

## Project Purpose
This project serves as a technical demonstration of:
- How to construct a clean data pipeline for sports analytics
- How to evaluate feature–target relationships in a noisy domain
- How to test predictive modeling assumptions using real data
- Why combine metrics, when used alone, fail to provide meaningful predictive accuracy
- How to integrate statistical analysis with interactive applications

The overall design provides a reusable structure for any combine-to-performance modeling problem or similar sports-centric analysis pipeline.


