# CombineProphetAnalytics

A data-driven analytics pipeline that integrates **NFL Combine performance data**, **NFL in-game tracking data**, and **Facebook Prophet time-series modeling** to evaluate player athleticism, project on-field performance, and generate coaching insights. This repository demonstrates how to combine traditionally separate data sources into a unified modeling framework that can replace or augment the NFL Combine with *game-verified, context-aware metrics*.

---

## Overview

This project ingests player Combine metrics, merges them with in-game Next Gen Stats tracking features, trains forecasting models using **Prophet**, and visualizes how physical traits relate to real in-game performance. The pipeline identifies:

- Trends and season-level changes in player performance  
- How Combine-like metrics evolve over time during real games  
- Athlete-specific projections (speed, acceleration, route tendencies, workload, etc.)
- Actionable coaching insights grounded in real movement data rather than lab-like Combine drills

---

## Key Features

### **1. Data Collection & Preprocessing**
- Loads **Combine datasets** (speed, agility, explosiveness metrics)
- Loads **in-game player tracking data** (speed, separation, acceleration, routes, distances)
- Performs:
  - Normalization  
  - Time alignment  
  - Player ID resolution  
  - Feature engineering  
  - Outlier filtering

### **2. Feature Engineering**
Creates modeling-ready variables:
- Rolling averages (3-game, 5-game, seasonal)
- Derived athleticism indicators
- Fatigue curves over time within games
- Burst/acceleration windows
- Route-type profiles (e.g., go routes vs. slants)
- Contextual features: down, distance, defensive shell

### **3. Prophet Time-Series Modeling**
For each player and feature:
- Builds univariate Prophet models for:
  - Peak speed over the season  
  - Separation trends  
  - Acceleration bursts  
  - Workload (routes/game)  
- Generates:
  - Forecasts  
  - Uncertainty intervals  
  - Seasonality graphs  
  - Changepoint detection

### **4. Combine ↔ Game Performance Alignment**
Maps Combine drills to in-game equivalents:

| Combine Metric | In-Game Analog |
|----------------|----------------|
| 40-yard dash | Top speed, burst speed windows |
| 20-shuttle | Short-area acceleration, change-of-direction tracking |
| Vertical jump | Explosiveness metrics in routes and breaks |
| 3-cone drill | Curvature & agility in routes |

This enables **context-aware athleticism scoring** that updates as the season progresses.

### **5. Visualizations**
Produces:
- Compare-and-contrast charts of Combine metrics vs. in-game data  
- Prophet forecast plots  
- Player performance trend dashboards  
- Efficiency vs. athleticism scatterplots  
- Radar plots of multi-metric performance  

---

## Example Insights

- Players with **high Combine speed** but **low in-game sustained speed** show fatigue or role-based reduction.
- Changepoints in acceleration forecasts often correlate with injury periods or role changes.
- In-game tracking can outperform Combine results in predicting:
  - YAC potential  
  - Route separation  
  - Break efficiency  
  - Defensive matchup outcomes  

The tool reveals *how the Combine fails to capture real performance stability*, and suggests using tracking-based metrics to supplement or replace Combine drills.

---

## Repository Structure

---

## Technical Stack

- **Python 3.x**
- **Prophet**
- **Pandas**, **NumPy**
- **Matplotlib**, **Plotly**
- **Scikit-learn** (for any auxiliary models) 
- **JupyterLab** for experimentation

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
2. **Prepare Data**
   - Place combine data in data/combine/
   - Place tracking data in: data/tracking/
3. **Run ETL + Feature Engineering**
4. **Run Prophet script**

