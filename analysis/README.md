# Analysis Scripts

## compare_predictions.py

This script compares **model predictions** (from trained Gaussian Process) with **true values** (from objective functions recorded in CSV).

### Usage

```bash
cd /home/fxl/MetalBayes
python analysis/compare_predictions.py
```

### What it does

1. **Loads optimization results**: Reads the most recent CSV file from `output/organic/`
   - CSV contains: input parameters and **true objective function values**
2. **Trains Gaussian Process model**: Uses all data points to train the surrogate model
3. **Gets model predictions**: Uses the trained model to predict on the same points
4. **Compares model vs truth**: Compares model predictions with true values from objective functions
5. **Generates comparison plots**:
   - **model_prediction_vs_truth_comparison.png**: Scatter plots showing model predictions vs true values for each objective (Adhesion, Coverage, Uniformity)
   - **model_residuals_plot.png**: Residual plots showing model prediction errors
   - **function_curves_comparison.png**: Function curves showing true functions vs model predictions in the parameter domain
     - Continuous true function curves
     - Continuous model prediction curves with uncertainty bands (±1σ)
     - Discrete training points marked
6. **Saves detailed comparison**: Creates `detailed_comparison.csv` with all model predictions, true values, and errors

### Output

All results are saved in `analysis/results/`:
- `model_prediction_vs_truth_comparison.png`: Scatter plots with R² and MAE metrics
- `model_residuals_plot.png`: Residual analysis plots
- `function_curves_comparison.png`: Function curves comparison (true functions vs model predictions)
  - Adhesion vs organic_ph
  - Coverage vs organic_soak_time
  - Uniformity vs organic_concentration
- `detailed_comparison.csv`: Detailed comparison table with model predictions and true values

### Metrics

For each objective, the script calculates:
- **R² Score**: Coefficient of determination (how well model fits the data)
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

### Note

- **CSV values** = True values from objective functions (ground truth)
- **Model predictions** = Predictions from trained Gaussian Process surrogate model
- This comparison shows how well the GP model approximates the true objective functions
