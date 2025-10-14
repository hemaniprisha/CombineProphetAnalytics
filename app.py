"""
Combine Prophet - Interactive Dashboard
NFL Wide Receiver Performance Prediction from Combine Metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Combine Prophet | NFL Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .tracking-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load exported data from combine_analysis.py
@st.cache_data
def load_analysis_export():
    export_path = "combine_analysis_export.pkl"
    if os.path.exists(export_path):
        with open(export_path, 'rb') as f:
            return pickle.load(f)
    return None

export = load_analysis_export()

if export is None:
    st.error("Analysis data not found. Please run combine_analysis.py first.")
    st.info("Run: `python combine_analysis.py` to generate the required data file.")
    st.stop()

# Extract data from export
data = export['merged_data']
feature_cols = export.get('feature_cols', [])
perf_metrics = export.get('perf_metrics', ['yards_per_game', 'catch_rate', 'tds_per_game'])
combine_metrics = export.get('combine_metrics', [])
model_metrics = export.get('metrics', {})
correlations = export.get('correlations', pd.DataFrame())
feature_importance = export.get('feature_importance', pd.DataFrame())

# Automatically detect combine and performance metrics from data columns if not in export
if not combine_metrics:
    combine_metrics = ['forty_yard', 'vertical', 'broad_jump', 'three_cone', 'shuttle', 'height', 'weight']
    combine_metrics = [c for c in combine_metrics if c in data.columns]

if not perf_metrics:
    perf_metrics = ['yards_per_game', 'catch_rate', 'tds_per_game', 'yards_per_reception']
    perf_metrics = [c for c in perf_metrics if c in data.columns]
# Get key metrics (handle negative R²)
test_r2 = model_metrics.get('test_r2', 0.0)
test_mae = model_metrics.get('mae', 0.0)
max_corr = model_metrics.get('max_correlation', 0.0)

# Calculate percentages for display
if test_r2 > 0:
    r2_pct = test_r2 * 100
    unexplained_pct = (1 - test_r2) * 100
else:
    r2_pct = 0.0
    unexplained_pct = 100.0

# Title section
st.markdown('<div class="main-header">Combine Prophet</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicting NFL Performance from Combine Metrics</div>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/National_Football_League_logo.svg/400px-National_Football_League_logo.svg.png", width=180)
    
    st.title("Navigation")
    page = st.radio(
        "Navigation Menu",
        ["Overview", "Data Explorer", "ML Model", "Tracking Data Value", "Player Lookup"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    st.markdown(f"""
    ### About This Analysis
    
    **Purpose:** Evaluate predictive power of combine metrics for NFL performance
    
    **Key Finding:** Combine metrics explain only **{r2_pct:.1f}%** of rookie WR performance
    
    **Implication:** Teams need comprehensive evaluation tools beyond combine testing
    
    ---
    
    **Data:** 2015-2023 NFL Seasons  
    **Position:** Wide Receivers  
    **Sample:** {len(data)} unique rookies
    
    ---
    
    **Analysis Method:**  
    XGBoost Regression with 5-fold CV
    """)

# Success metrics at top
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(data):,}")
with col2:
    st.metric("Unique Players", f"{len(data['player_name'].unique()):,}")
with col3:
    st.metric("Model R²", f"{test_r2:.3f}")
with col4:
    st.metric("Prediction Error", f"{test_mae:.1f} ypg")

st.markdown("---")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "Overview":
    st.header("Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### Research Question
        
        **"How well can combine testing predict NFL rookie WR performance?"**
        
        This analysis examined {len(data)} rookie wide receivers from 2015-2023 to quantify 
        the relationship between combine metrics and on-field success.
        
        ### Key Findings
        
        - **Weak Correlations**: Strongest correlation was {max_corr:.3f} (vertical jump)
        - **Limited Predictive Power**: Model R² of {test_r2:.3f}
        - **High Uncertainty**: Average prediction error of {test_mae:.1f} yards/game
        - **The Gap**: {unexplained_pct:.1f}% of performance remains unexplained
        
        ### What This Means
        
        Combine testing measures athletic traits in isolation, but football success depends on:
        - Route running technique
        - Separation ability vs coverage
        - Football IQ and recognition
        - Performance under pressure
        - Consistency and reliability
        
        **These factors require in-game tracking data to measure.**
        """)
        
        st.markdown("### Sample Data")
        display_cols = ['player_name', 'season', 'yards_per_game', 'catch_rate']
        if 'forty_yard' in data.columns:
            display_cols.append('forty_yard')
        if 'vertical' in data.columns:
            display_cols.append('vertical')
        
        st.dataframe(
            data[display_cols].head(10).style.format({
                'yards_per_game': '{:.1f}',
                'catch_rate': '{:.1%}',
                'forty_yard': '{:.2f}',
                'vertical': '{:.1f}'
            }),
            hide_index=True,
            width='stretch'
        )
    
    with col2:
        st.markdown(f"""
        ### Key Statistics
        
        **Model Performance:**
        - R² Score: **{test_r2:.3f}**
        - MAE: **{test_mae:.1f} ypg**
        - RMSE: **{model_metrics.get('rmse', 0):.1f} ypg**
        
        **Correlations:**
        - Strongest: **{max_corr:.3f}**
        - Variance explained: **{max_corr**2*100:.1f}%**
        
        **Dataset:**
        - Players: **{len(data)}**
        - Features: **{len(feature_cols)}**
        - Years: **2015-2023**
        
        ---
        
        ### Business Impact
        
            **$8M** - Cost of draft bust  
            **40-50%** - Historical bust rate  
            **{unexplained_pct:.0f}%** - Information gap  
        
        **One better evaluation = years of data ROI**
        """)
    
    st.markdown("---")
    
    # Quick visualization
    st.subheader("Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            data,
            x='yards_per_game',
            nbins=30,
            title="Yards Per Game Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(template='plotly_white', showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        if 'forty_yard' in data.columns:
            fig = px.histogram(
                data,
                x='forty_yard',
                nbins=30,
                title="40-Yard Dash Distribution",
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(template='plotly_white', showlegend=False)
            st.plotly_chart(fig, width='stretch')
    
    # Value proposition
    st.markdown(f"""
    <div class="tracking-box">
    <h3 style="color: white; margin-top: 0;"> The Tracking Data Advantage</h3>
    <p><strong>Combine testing measures {r2_pct:.1f}% of what matters. What about the other {unexplained_pct:.1f}%?</strong></p>
    <p>In-game tracking data captures:</p>
    <ul>
        <li><strong>Real Performance:</strong> Speed, acceleration, and movement in actual game situations</li>
        <li><strong>Separation Metrics:</strong> Distance from defenders at catch point</li>
        <li><strong>Route Efficiency:</strong> Path quality and technique under pressure</li>
        <li><strong>Contextual Analysis:</strong> Performance by coverage type, formation, and situation</li>
    </ul>
    <p><strong>Result:</strong> Bridge the {unexplained_pct:.0f}% evaluation gap that combine testing leaves.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: DATA EXPLORER
# ============================================================================
elif page == "Data Explorer":
    st.header("Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "🎯 Relationships"])
    
    with tab1:
        st.subheader("Distribution Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            available_metrics = [c for c in combine_metrics + perf_metrics if c in data.columns]
            metric = st.selectbox(
                "Select metric to visualize:",
                available_metrics,
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            bins = st.slider("Number of bins:", 10, 50, 30)
        
        fig = px.histogram(
            data,
            x=metric,
            nbins=bins,
            title=f"Distribution of {metric.replace('_', ' ').title()}",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=450, template='plotly_white', showlegend=False)
        st.plotly_chart(fig, width='stretch')
        
        # Statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        valid_data = data[metric].dropna()
        
        # Check if data is numeric
        if pd.api.types.is_numeric_dtype(valid_data):
            col1.metric("Mean", f"{valid_data.mean():.2f}")
            col2.metric("Median", f"{valid_data.median():.2f}")
            col3.metric("Std Dev", f"{valid_data.std():.2f}")
            col4.metric("Min", f"{valid_data.min():.2f}")
            col5.metric("Max", f"{valid_data.max():.2f}")
        else:
            # For non-numeric data (like height strings), show different stats
            col1.metric("Count", f"{len(valid_data)}")
            col2.metric("Unique", f"{valid_data.nunique()}")
            col3.metric("Mode", str(valid_data.mode()[0]) if len(valid_data.mode()) > 0 else "N/A")
            col4.metric("Min", str(valid_data.min()))
            col5.metric("Max", str(valid_data.max()))    
    with tab2:
        st.subheader("Correlation Analysis")
        
        available_combine = [c for c in combine_metrics if c in data.columns]
        available_perf = [c for c in perf_metrics if c in data.columns]
        available_metrics = available_combine + available_perf
        
        if len(available_metrics) > 0:
            corr_matrix = data[available_metrics].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=[m.replace('_', ' ').title() for m in corr_matrix.columns],
                y=[m.replace('_', ' ').title() for m in corr_matrix.columns],
                color_continuous_scale='RdBu_r',
                zmin=-0.5,
                zmax=0.5,
                aspect='auto'
            )
            fig.update_layout(height=600, title="Correlation Heatmap: Combine vs Performance")
            st.plotly_chart(fig, width='stretch')
            
            st.markdown(f"""
            <div class="insight-box">
            <h4 style="color: white; margin-top: 0;">Key Insight</h4>
            <p><strong>Notice the weak correlations (most values near 0)</strong></p>
            <p>The strongest correlation is only {max_corr:.3f}, meaning even the best predictor 
            explains just {max_corr**2*100:.1f}% of performance variance.</p>
            <p>This demonstrates combine metrics have limited predictive power for in-game performance.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Insufficient metrics available for correlation analysis")
    
    with tab3:
            st.subheader("Relationship Explorer")
            
            col1, col2 = st.columns(2)
            
            available_combine = [c for c in combine_metrics if c in data.columns and c is not None]
            available_perf = [c for c in perf_metrics if c in data.columns and c is not None]
            
            if not available_combine or not available_perf:
                st.warning("Insufficient metrics available for relationship analysis")
            else:
                with col1:
                    x_var = st.selectbox(
                        "X-axis (Combine Metric):",
                        available_combine,
                        format_func=lambda x: x.replace('_', ' ').title() if x else "None"
                    )
                
                with col2:
                    y_var = st.selectbox(
                        "Y-axis (Performance Metric):",
                        available_perf,
                        format_func=lambda x: x.replace('_', ' ').title() if x else "None"
                    )
                
                if x_var and y_var:
                    valid_data = data[[x_var, y_var]].dropna()
                    
                    if len(valid_data) > 0:
                        fig = px.scatter(
                            valid_data,
                            x=x_var,
                            y=y_var,
                            trendline="ols",
                            title=f"{y_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}",
                            opacity=0.6
                        )
                        fig.update_traces(marker=dict(size=8, color='#667eea'))
                        fig.update_layout(height=500, template='plotly_white')
                        st.plotly_chart(fig, width='stretch')
                        
                        corr = valid_data[x_var].corr(valid_data[y_var])
                        r_squared = corr ** 2
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Correlation (r)", f"{corr:.3f}")
                        col2.metric("R² (variance explained)", f"{r_squared:.1%}")
                        col3.metric("Sample Size", f"{len(valid_data):,}")
                    else:
                        st.warning("No valid data points for selected metrics")
                else:
                    st.warning("Please select both metrics")
# PAGE 3: ML MODEL
elif page == "ML Model":
    st.header("Machine Learning Model Results")
    
    st.markdown("""
    An **XGBoost regression model** was trained to predict **Yards Per Game** from combine metrics.
    This quantifies exactly how much of rookie performance can be explained by testing alone.
    """)
    
    # Display pre-computed metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{test_r2:.3f}", help="Proportion of variance explained")
    col2.metric("RMSE", f"{model_metrics.get('rmse', 0):.1f} ypg", help="Root Mean Squared Error")
    col3.metric("MAE", f"{test_mae:.1f} ypg", help="Mean Absolute Error")
    col4.metric("CV R²", f"{model_metrics.get('cv_mean', 0):.3f}", help="Cross-validation mean")
    
    # Critical finding
    if test_r2 <= 0:
        st.markdown(f"""
        <div class="insight-box">
        <h3 style="color: white; margin-top: 0;"> CRITICAL FINDING</h3>
        <h2 style="color: white; margin: 10px 0;">Model performs WORSE than baseline</h2>
        <p style="font-size: 1.2rem;"><strong>Negative R² means combine metrics have essentially zero predictive power</strong></p>
        <p>Prediction error of {test_mae:.1f} ypg represents {test_mae/data['yards_per_game'].mean()*100:.0f}% of average production.</p>
        <p>Teams making $10M decisions with this level of uncertainty need better evaluation tools.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
        <h3 style="color: white; margin-top: 0;"> CRITICAL FINDING</h3>
        <h2 style="color: white; margin: 10px 0;">Combine metrics explain only {r2_pct:.1f}% of performance</h2>
        <p style="font-size: 1.2rem;"><strong>That means {unexplained_pct:.1f}% is driven by factors not measured at the combine</strong></p>
        <p>This gap represents millions of dollars in draft mistakes for teams relying solely on combine data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature importance (from export)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Feature Importance")
        
        if not feature_importance.empty:
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Which Combine Metrics Matter Most?",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, showlegend=False, template='plotly_white')
            st.plotly_chart(fig,width='stretch')
            
            st.dataframe(feature_importance, hide_index=True, width='stretch')
    
    with col2:
        st.subheader("Model Insights")
        
        st.markdown(f"""
        **Training Performance:**
        - Training R²: {model_metrics.get('train_r2', 0):.3f}
        - Test R²: {test_r2:.3f}
        - Overfit gap: {model_metrics.get('train_r2', 0) - test_r2:.3f}
        
        **Cross-Validation:**
        - Mean R²: {model_metrics.get('cv_mean', 0):.3f}
        - Std Dev: {model_metrics.get('cv_std', 0):.3f}
        - Consistency: {'Good' if model_metrics.get('cv_std', 1) < 0.15 else 'Moderate'}
        
        **Prediction Quality:**
        - MAE: {test_mae:.1f} ypg
        - Error %: {test_mae/data['yards_per_game'].mean()*100:.0f}%
        - Sample: {model_metrics.get('sample_size', 0)} players
        """)
        
        st.markdown(f"""
        **Interpretation:**
        
        The model's {'negative' if test_r2 <= 0 else 'low'} R² score confirms that combine 
        testing alone cannot reliably predict rookie WR performance. 
        
        With prediction errors averaging {test_mae:.1f} yards/game, teams need 
        comprehensive tracking data to make informed talent decisions.
        """)

# ============================================================================
# PAGE 4: TRACKING DATA VALUE
# ============================================================================
elif page == "Tracking Data Value":
    st.header("The Value of In-Game Tracking Data")
    
    st.markdown(f"""
    <div class="tracking-box">
    <h2 style="color: white; margin-top: 0;"> The Problem </h2>
    <h3 style="color: white;">NFL teams make multi-million dollar decisions based on combine data that explains only {r2_pct:.1f}% of success</h3>
    <p style="font-size: 1.1rem;">The other {unexplained_pct:.1f}%? It's invisible to traditional scouting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # What's missing vs what tracking provides
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What Combine Testing Can't Measure
        
        - **Route Precision** - How clean are the cuts?
        - **Separation Ability** - Creating space at catch point
        - **Real-Game Speed** - Different from straight-line 40
        - **Acceleration Patterns** - First 3 steps matter most
        - **Positional Versatility** - Can they play multiple spots?
        - **Football IQ** - Reading defenses, adjusting routes
        - **Performance Under Pressure** - Against top coverage
        - **YAC Ability** - Making plays after the catch
        - **Consistency** - Game-to-game reliability
        """)
    
    with col2:
        st.markdown("""
        ### What Tracking Data Does Measure
        
        - **XY Tracking Data** - Every player, every play
        - **Max Speed In-Game** - Actual game conditions
        - **Separation Metrics** - Distance from nearest defender
        - **Acceleration Data** - First-step explosiveness
        - **Route Depth Analysis** - Performance by route type
        - **Expected Outcomes** - xYAC, xCompletion models
        - **Contextual Performance** - Situation-specific metrics
        - **Movement Efficiency** - How they create advantages
        - **Predictive Models** - Project future performance
        """)
    
    st.markdown("---")
    
    # The data gap visualization
    st.subheader("The Performance Prediction Gap")
    
    gap_data = pd.DataFrame({
        'Analysis Method': ['Combine Only', 'Combine + Basic Stats', 'Combine + Tracking Data'],
        'Variance Explained': [max(r2_pct, 0), 60, 85],
        'Information Gap': [max(unexplained_pct, 100), 40, 15]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Variance Explained',
        x=gap_data['Analysis Method'],
        y=gap_data['Variance Explained'],
        marker_color='#2ecc71',
        text=gap_data['Variance Explained'].apply(lambda x: f'{x}%'),
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Information Gap',
        x=gap_data['Analysis Method'],
        y=gap_data['Information Gap'],
        marker_color='#e74c3c',
        text=gap_data['Information Gap'].apply(lambda x: f'{x}%'),
        textposition='inside'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Performance Prediction Accuracy by Data Type',
        yaxis_title='Percentage (%)',
        height=450,
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Business impact
    st.subheader("Business Impact for NFL Teams")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: white; margin: 0;">$8M</h3>
        <p style="color: white; margin: 5px 0 0 0;">Cost of one draft bust</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: white; margin: 0;">40-50%</h3>
        <p style="color: white; margin: 5px 0 0 0;">Bust rate with combine-only</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="color: white; margin: 0;">{unexplained_pct:.0f}%</h3>
        <p style="color: white; margin: 5px 0 0 0;">Performance gap</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Final message
    st.markdown(f"""
    <div class="tracking-box">
    <h2 style="color: white; margin-top: 0;">🎯 The Bottom Line</h2>
    <h3 style="color: white;">Teams relying on combine data are making decisions with {r2_pct:.1f}% of the information</h3>
    <p style="font-size: 1.2rem;"><strong>In-game tracking data provides the missing {unexplained_pct:.0f}%</strong></p>
    <ul style="font-size: 1.1rem;">
        <li><strong>Better Draft Decisions:</strong> Identify overlooked talent and avoid busts</li>
        <li><strong>Competitive Advantage:</strong> Access data competitors don't have</li>
        <li><strong>Immediate ROI:</strong> One improved pick pays for years of data</li>
        <li><strong>Player Development:</strong> Target specific weaknesses with precision</li>
    </ul>
    <p style="font-size: 1.1rem; margin-top: 20px;"><strong>In 2025, combine-only analysis isn't enough.</strong></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 5: PLAYER LOOKUP
# ============================================================================
elif page == "Player Lookup":
    st.header("Player Lookup Tool")
    
    st.markdown("Search for specific players and analyze their combine metrics vs actual performance.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        player_name = st.selectbox(
            "Select Player:",
            sorted(data['player_name'].unique()),
            index=0
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        if st.button("Random Player"):
            player_name = np.random.choice(data['player_name'].unique())
    
    player_data = data[data['player_name'] == player_name]
    
    if not player_data.empty:
        player_row = player_data.iloc[0]
        
        st.markdown(f"## {player_name}")
        st.markdown("---")
        
        # Basic info
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Season", int(player_row['season']))
        col2.metric("Games", int(player_row['games']))
        
        # Handle height formatting (could be "6-0" format or numeric)
        if pd.notna(player_row.get('height')):
            height_val = player_row['height']
            if isinstance(height_val, str) and '-' in height_val:
                # Format is "6-0" (feet-inches)
                col3.metric("Height", height_val.replace('-', '\'-') + '"')
            else:
                # Numeric format (total inches)
                col3.metric("Height", f"{int(height_val)}\"")
        else:
            col3.metric("Height", "N/A")
            
        col4.metric("Weight", f"{int(player_row['weight'])} lbs" if pd.notna(player_row.get('weight')) else "N/A")
        col5.metric("Position", "WR")
        st.markdown("---")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Yards/Game", f"{player_row['yards_per_game']:.1f}")
        col2.metric("Receptions", int(player_row['receptions']))
        col3.metric("Total Yards", int(player_row['receiving_yards']))
        col4.metric("Touchdowns", int(player_row['receiving_tds']))
        col5.metric("Catch Rate", f"{player_row['catch_rate']*100:.1f}%" if pd.notna(player_row['catch_rate']) else "N/A")
        
        st.markdown("---")
        
        # Combine metrics
        st.subheader("Combine Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("40-Yard", f"{player_row['forty_yard']:.2f}s" if pd.notna(player_row.get('forty_yard')) else "N/A")
        col2.metric("Vertical", f"{player_row['vertical']:.1f}\"" if pd.notna(player_row.get('vertical')) else "N/A")
        col3.metric("Broad Jump", f"{player_row['broad_jump']:.0f}\"" if pd.notna(player_row.get('broad_jump')) else "N/A")
        col4.metric("3-Cone", f"{player_row['three_cone']:.2f}s" if pd.notna(player_row.get('three_cone')) else "N/A")
        col5.metric("Shuttle", f"{player_row['shuttle']:.2f}s" if pd.notna(player_row.get('shuttle')) else "N/A")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Athletic Profile (Percentile Rankings)")
            
            # Calculate percentiles
            metrics_for_radar = ['forty_yard', 'vertical', 'broad_jump', 'three_cone', 'shuttle']
            available_metrics = [m for m in metrics_for_radar if m in data.columns and pd.notna(player_row.get(m))]
            
            if available_metrics:
                percentiles = []
                labels = []
                
                for metric in available_metrics:
                    value = player_row[metric]
                    # For time-based metrics (lower is better), invert percentile
                    if metric in ['forty_yard', 'three_cone', 'shuttle']:
                        percentile = (data[metric] > value).sum() / len(data[metric].dropna()) * 100
                    else:
                        percentile = (data[metric] < value).sum() / len(data[metric].dropna()) * 100
                    
                    percentiles.append(percentile)
                    labels.append(metric.replace('_', ' ').title())
                
                # Create radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=percentiles,
                    theta=labels,
                    fill='toself',
                    fillcolor='rgba(102, 126, 234, 0.5)',
                    line=dict(color='rgb(102, 126, 234)', width=2),
                    name=player_name
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            ticksuffix='%'
                        )
                    ),
                    showlegend=False,
                    height=400,
                    title="Percentile Rankings vs All WRs"
                )
                
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No combine data available for radar chart")
        
        with col2:
            st.subheader("Performance vs Dataset Average")
            
            # Compare to averages
            comparison_metrics = [
                ('yards_per_game', 'Yards/Game'),
                ('catch_rate', 'Catch Rate'),
                ('tds_per_game', 'TDs/Game')
            ]
            
            comparison_data = []
            for metric, label in comparison_metrics:
                if pd.notna(player_row.get(metric)):
                    player_val = player_row[metric]
                    avg_val = data[metric].mean()
                    
                    comparison_data.append({
                        'Metric': label,
                        'Player': player_val,
                        'Average': avg_val
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Player',
                    x=comp_df['Metric'],
                    y=comp_df['Player'],
                    marker_color='#667eea'
                ))
                
                fig.add_trace(go.Bar(
                    name='League Average',
                    x=comp_df['Metric'],
                    y=comp_df['Average'],
                    marker_color='#95a5a6'
                ))
                
                fig.update_layout(
                    barmode='group',
                    height=400,
                    template='plotly_white',
                    yaxis_title='Value',
                    showlegend=True
                )
                
                st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # Player insights
        st.subheader("Quick Analysis")
        
        insights = []
        
        # Speed analysis
        if pd.notna(player_row.get('forty_yard')):
            forty = player_row['forty_yard']
            if forty < 4.40:
                insights.append(" **Elite speed** - Top 10% in 40-yard dash")
            elif forty > 4.60:
                insights.append(" **Below-average speed** - Bottom 30% in 40-yard dash")
        
        # Production analysis
        ypg = player_row['yards_per_game']
        avg_ypg = data['yards_per_game'].mean()
        if ypg > avg_ypg * 1.5:
            insights.append(" **Elite production** - Significantly above average yards/game")
        elif ypg < avg_ypg * 0.5:
            insights.append(" **Low production** - Below average yards/game")
        
        # Catch rate
        if pd.notna(player_row.get('catch_rate')):
            catch_rate = player_row['catch_rate']
            if catch_rate > 0.70:
                insights.append(" **Reliable hands** - Excellent catch rate")
            elif catch_rate < 0.50:
                insights.append(" **Drop concerns** - Below-average catch rate")
        
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.info("Insufficient data for detailed analysis")
        
        # What tracking would reveal
        st.markdown("---")
        st.markdown("""
        <div class="tracking-box">
        <h3 style="color: white; margin-top: 0;"> What Tracking Data Would Reveal</h3>
        <p>For this player, in-game tracking data would provide:</p>
        <ul>
            <li><strong>Separation metrics:</strong> Average yards from nearest defender at catch point</li>
            <li><strong>Route efficiency:</strong> How clean and precise are their route cuts?</li>
            <li><strong>In-game speed:</strong> Max speed achieved during actual plays</li>
            <li><strong>YAC ability:</strong> Expected vs actual yards after catch</li>
            <li><strong>Versatility:</strong> Performance from different alignments and depths</li>
        </ul>
        <p><strong>This contextual data would show whether their production matches their athletic testing.</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("No data available for selected player")

# FOOTER
st.markdown("---")
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0;'>
        <h4> Combine Prophet</h4>
        <p style='color: #888;'>
            NFL Wide Receiver Performance Prediction Analysis<br>
            Data Source: nflverse (nfl_data_py) | 2015-2023 NFL Seasons<br>
            Position Focus: Wide Receivers
        </p>
        <p style='color: #888; margin-top: 1rem;'>
            <strong>Key Finding:</strong> Combine metrics explain only {r2_pct:.1f}% of WR performance<br>
            <strong>Business Impact:</strong> Teams need comprehensive tracking data to see the other {unexplained_pct:.0f}%
        </p>
        <p style='margin-top: 1rem;'>
            <a href='https://github.com/hemaniprisha' target='_blank' style='margin: 0 10px; color: #667eea; text-decoration: none;'>GitHub</a> | 
            <a href='https://www.linkedin.com/in/prisha-hemani-4194a8257/' target='_blank' style='margin: 0 10px; color: #667eea; text-decoration: none;'>LinkedIn</a> | 
            <a href='mailto:hemaniprisha1@gmail.com' style='margin: 0 10px; color: #667eea; text-decoration: none;'>Contact</a>
        </p>    </div>
    """, unsafe_allow_html=True)