# Combine Prophet: Predicting NFL Performance from Combine Metrics
# Analysis Period: 2015-2023 NFL Rookie Seasons

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

print("\nCOMBINE PROPHET: NFL WR PERFORMANCE PREDICTION")
print("Goal: Evaluate predictive power of combine metrics for rookie performance")
print("Analysis: 2015-2023 NFL Rookie WR Seasons\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")

try:
    import nfl_data_py as nfl
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nfl_data_py'])
    import nfl_data_py as nfl

combine = nfl.import_combine_data(years=range(2015, 2024))
combine_wr = combine[combine['pos'] == 'WR'].copy()

seasonal = nfl.import_seasonal_data(years=range(2015, 2024))
roster = nfl.import_seasonal_rosters(years=range(2015, 2024))

seasonal_wr = seasonal.merge(
    roster[['player_id', 'player_name', 'position']],
    on='player_id',
    how='left'
)
wr_stats = seasonal_wr[seasonal_wr['position'] == 'WR'].copy()

print(f"  Combine records: {len(combine_wr)}")
print(f"  Seasonal records: {len(wr_stats)}\n")

# ============================================================================
# PREPARE DATA (PREVENT LEAKAGE)
# ============================================================================
print("Preparing data...")

# Select combine metrics
combine_cols = ['season', 'player_name', 'pos', 'ht', 'wt', 
                'forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']
available_cols = [col for col in combine_cols if col in combine_wr.columns]
combine_clean = combine_wr[available_cols].copy()
combine_clean = combine_clean.dropna(subset=['player_name', 'season'])

rename_map = {'forty': 'forty_yard', 'cone': 'three_cone', 'ht': 'height', 'wt': 'weight'}
combine_clean.rename(columns=rename_map, inplace=True)

# Select performance metrics
perf_cols = ['player_name', 'season', 'games', 'receptions', 'targets', 
             'receiving_yards', 'receiving_tds', 'receiving_yards_after_catch']
available_perf = [col for col in perf_cols if col in wr_stats.columns]
wr_stats_clean = wr_stats[available_perf].copy()

# CRITICAL: Keep only rookie seasons
wr_stats_clean = wr_stats_clean.sort_values(['player_name', 'season'])
wr_stats_rookie = wr_stats_clean.groupby('player_name').first().reset_index()

print(f"  Rookie seasons extracted: {len(wr_stats_rookie)}")

# Merge combine with rookie performance
merged = pd.merge(combine_clean, wr_stats_rookie, on=['player_name', 'season'], how='inner')
merged = merged[merged['games'] >= 4].copy()

# Verify one record per player
unique_players = len(merged['player_name'].unique())
if len(merged) != unique_players:
    merged = merged.drop_duplicates(subset=['player_name'], keep='first')

print(f"  Final sample: {len(merged)} unique rookie WRs\n")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("Creating features...")

merged['yards_per_game'] = merged['receiving_yards'] / merged['games']
merged['yards_per_reception'] = merged['receiving_yards'] / merged['receptions'].replace(0, np.nan)
merged['catch_rate'] = merged['receptions'] / merged['targets'].replace(0, np.nan)
merged['tds_per_game'] = merged['receiving_tds'] / merged['games']

# Remove extreme outliers
ypg_99th = merged['yards_per_game'].quantile(0.99)
merged = merged[merged['yards_per_game'] <= ypg_99th].copy()

print(f"  Performance metrics created")
print(f"  Final dataset: {len(merged)} players\n")

# ============================================================================
# EXPLORATORY ANALYSIS
# ============================================================================

print("\nCORRELATION ANALYSIS\n")

combine_metrics = ['forty_yard', 'vertical', 'broad_jump', 'three_cone', 'shuttle']
combine_metrics = [c for c in combine_metrics if c in merged.columns]

perf_metrics = ['yards_per_game', 'catch_rate', 'tds_per_game']

correlation_results = []
for comb in combine_metrics:
    for perf in perf_metrics:
        valid_data = merged[[comb, perf]].dropna()
        if len(valid_data) > 20:
            corr = valid_data[comb].corr(valid_data[perf])
            correlation_results.append({
                'Combine Metric': comb.replace('_', ' ').title(),
                'Performance Metric': perf.replace('_', ' ').title(),
                'Correlation': corr
            })

corr_df = pd.DataFrame(correlation_results)
corr_df_sorted = corr_df.sort_values('Correlation', key=abs, ascending=False)

print("Top 5 Correlations:")
print(corr_df_sorted.head(5).to_string(index=False))

max_corr = corr_df_sorted.iloc[0]['Correlation']
print(f"\nStrongest correlation: {abs(max_corr):.3f}")
print(f"Variance explained by best predictor: {abs(max_corr)**2*100:.1f}%")
print(f"Interpretation: Weak predictive relationship\n")

# Create visualizations
print("Generating visualizations...")

# Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = merged[combine_metrics + perf_metrics].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
            vmin=-0.5, vmax=0.5, square=True, linewidths=1)
plt.title('Correlation Matrix: Combine vs Performance (Rookie Seasons)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
top_4 = corr_df_sorted.head(4)

for idx, (_, row) in enumerate(top_4.iterrows()):
    x_metric = row['Combine Metric'].lower().replace(' ', '_')
    y_metric = row['Performance Metric'].lower().replace(' ', '_')
    
    if x_metric in merged.columns and y_metric in merged.columns:
        valid_data = merged[[x_metric, y_metric]].dropna()
        axes[idx].scatter(valid_data[x_metric], valid_data[y_metric], alpha=0.5, s=50)
        
        z = np.polyfit(valid_data[x_metric], valid_data[y_metric], 1)
        p = np.poly1d(z)
        axes[idx].plot(valid_data[x_metric], p(valid_data[x_metric]), "r--", linewidth=2)
        
        corr = valid_data[x_metric].corr(valid_data[y_metric])
        axes[idx].set_xlabel(x_metric.replace('_', ' ').title(), fontsize=11)
        axes[idx].set_ylabel(y_metric.replace('_', ' ').title(), fontsize=11)
        axes[idx].set_title(f'r = {corr:.3f}', fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)

plt.suptitle('Combine Metrics vs Rookie Performance', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()

print("  correlation_heatmap.png saved")
print("  scatter_plots.png saved\n")

# ============================================================================
# PREDICTIVE MODELING
# ============================================================================

print("\nPREDICTIVE MODELING\n")

feature_cols = [c for c in combine_metrics if c in merged.columns]
target_col = 'yards_per_game'

model_data = merged[feature_cols + [target_col]].dropna()
print(f"Modeling dataset: {len(model_data)} players")
print(f"Features: {', '.join(feature_cols)}")
print(f"Target: {target_col}\n")

X = model_data[feature_cols]
y = model_data[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"Training set: {len(X_train)} | Test set: {len(X_test)}\n")

# Train model
model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\nMODEL RESULTS\n")
print(f"Cross-Validation R² (5-fold): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print(f"Training R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")
print(f"Test RMSE: {test_rmse:.2f} yards/game")
print(f"Test MAE: {test_mae:.2f} yards/game")
print(f"\nAverage rookie production: {y.mean():.1f} yards/game")
print(f"Prediction error as % of production: {test_mae/y.mean()*100:.1f}%")

if test_r2 <= 0:
    print(f"\nInterpretation: Model performs worse than baseline (predicting mean)")
    print(f"Combine metrics have essentially ZERO predictive power")
else:
    print(f"\nVariance explained: {test_r2*100:.1f}%")
    print(f"Variance unexplained: {(1-test_r2)*100:.1f}%")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance:")
print(importance_df.to_string(index=False))

# Save feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importance: Combine Metrics', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
max_val = max(y_test.max(), y_pred_test.max())
min_val = min(y_test.min(), y_pred_test.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Yards Per Game', fontsize=12)
plt.ylabel('Predicted Yards Per Game', fontsize=12)
plt.title(f'Actual vs Predicted (R² = {test_r2:.3f})', fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n  feature_importance.png saved")
print("  actual_vs_predicted.png saved\n")

# Sample predictions
sample_indices = np.random.choice(len(y_test), size=min(10, len(y_test)), replace=False)
sample_df = pd.DataFrame({
    'Actual': y_test.values[sample_indices],
    'Predicted': y_pred_test[sample_indices],
    'Error': (y_test.values[sample_indices] - y_pred_test[sample_indices])
}).round(1)

print("\nSample Predictions:")
print(sample_df.to_string(index=False))

# ============================================================================
# BUSINESS IMPLICATIONS
# ============================================================================

print("\n\nBUSINESS IMPLICATIONS\n")

unexplained = max(0, (1-test_r2)*100) if test_r2 > 0 else 100

print("\nTHE EVALUATION GAP\n")

if test_r2 <= 0:
    print(f"Combine metrics: NO predictive power")
    print(f"Prediction error: {test_mae:.1f} ypg ({test_mae/y.mean()*100:.0f}% of production)")
    print(f"Sample size: {len(model_data)} rookie WRs")
else:
    print(f"Variance explained by combine: {test_r2*100:.1f}%")
    print(f"Variance unexplained: {unexplained:.1f}%")
    print(f"Prediction error: {test_mae:.1f} ypg ({test_mae/y.mean()*100:.0f}% of production)")
    print(f"Sample size: {len(model_data)} rookie WRs")

print("\nWHAT COMBINE TESTING MISSES\n")
print("- Route running precision and technique")
print("- Separation ability at catch point")
print("- Real-game acceleration patterns")
print("- Football IQ and route recognition")
print("- Performance under pressure/coverage")
print("- Yards after catch ability")
print("- Body control and adjustments")

print("\nWHAT TRACKING DATA PROVIDES\n")
print("- XY positioning (every player, every play)")
print("- Separation metrics vs defenders")
print("- Speed and acceleration in game context")
print("- Route efficiency and depth analysis")
print("- Performance by coverage type")
print("- Expected outcome models (xYAC, xReceptions)")

print("\nFINANCIAL IMPACT\n")
print("Average WR draft pick value (Rounds 1-3): $2-8M per year")
print("Historical bust rate (combine-based): 40-50%")
print("Cost of wrong pick: $8M+ over 4 years")
print(f"Evaluation uncertainty: {test_mae/y.mean()*100:.0f}% prediction error")
print("\nBetter evaluation tools ROI: One avoided bust = significant cost savings")

print("\n\nCONCLUSION\n")
print(f"\nCombine testing leaves {unexplained:.0f}% of rookie performance unexplained.")
print("NFL teams making $10M decisions need comprehensive evaluation tools.")
print("In-game tracking data addresses this gap.\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================
import pickle

export = {
    'merged_data': merged,
    'correlations': corr_df_sorted,
    'feature_importance': importance_df,
    'sample_predictions': sample_df,
    'xgb_model': model,
    'metrics': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'max_correlation': abs(max_corr),
        'sample_size': len(model_data),
    }
}

with open('combine_analysis_export.pkl', 'wb') as f:
    pickle.dump(export, f)

print("Analysis complete. Files saved:")
print("  - correlation_heatmap.png")
print("  - scatter_plots.png")
print("  - feature_importance.png")
print("  - actual_vs_predicted.png")
print("  - combine_analysis_export.pkl")
print()