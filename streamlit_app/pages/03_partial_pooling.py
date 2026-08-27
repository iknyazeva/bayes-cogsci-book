import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Partial Pooling & Shrinkage", page_icon="🌲", layout="wide")

st.title("🌲 Module 3: Multilevel Models, Partial Pooling & Shrinkage")
st.markdown(r"""
*Course: Bayesian Analysis of Empirical Data (Chapter 10)*  
Visualize how hierarchical / multilevel models balance individual group estimates with the population mean through **adaptive shrinkage**.
""")

with st.expander("📖 Generative Model & Shrinkage Formula", expanded=False):
    st.markdown(r"""
    **Two-Level Gaussian Hierarchy:**
    $$\theta_j \sim \mathcal{N}(\mu, \tau^2) \quad \text{(Group Means)}$$
    $$y_{ij} \sim \mathcal{N}(\theta_j, \sigma^2) \quad \text{(Individual Observations within Group } j)$$
    
    **Analytical Shrinkage Weight:**
    $$\hat{\theta}_j = w_j \bar{y}_j + (1 - w_j) \mu \quad \text{where} \quad w_j = \frac{\tau^2}{\tau^2 + \sigma^2 / n_j}$$
    - **Small group ($n_j$ small)** $\implies w_j \to 0 \implies \hat{\theta}_j \to \mu$ (strong shrinkage toward grand mean).
    - **Large group ($n_j$ large)** $\implies w_j \to 1 \implies \hat{\theta}_j \to \bar{y}_j$ (data dominates).
    """)

st.sidebar.header("⚙️ Hierarchy Parameters")
grand_mu = st.sidebar.slider(r"Population Grand Mean $\mu$", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
tau = st.sidebar.slider(r"Between-Group SD $\tau$", min_value=1.0, max_value=25.0, value=8.0, step=0.5, help="Heterogeneity across groups")
sigma = st.sidebar.slider(r"Within-Group Noise $\sigma$", min_value=1.0, max_value=30.0, value=12.0, step=0.5, help="Individual measurement variation")

n_groups = st.sidebar.slider("Number of Groups ($J$)", min_value=4, max_value=20, value=8, step=1)
seed = st.sidebar.number_input("Random Data Seed", value=42, step=1)
np.random.seed(seed)

# Generate synthetic groups with varying sample sizes
# Ensure some groups are small (n=3) and some large (n=60)
sample_sizes = np.random.choice([3, 5, 8, 15, 30, 60], size=n_groups)
true_thetas = np.random.normal(grand_mu, tau, size=n_groups)

# Generate observed group means
observed_means = np.zeros(n_groups)
for j in range(n_groups):
    y_obs = np.random.normal(true_thetas[j], sigma, size=sample_sizes[j])
    observed_means[j] = np.mean(y_obs)

# Compute weights and partial pooling estimates
weights = (tau**2) / (tau**2 + (sigma**2) / sample_sizes)
partial_pooled = weights * observed_means + (1.0 - weights) * grand_mu
shrinkage_pct = (1.0 - weights) * 100.0

col1, col2 = st.columns([0.65, 0.35])

with col1:
    fig = go.Figure()
    
    # Grand mean line
    fig.add_hline(y=grand_mu, line_dash="dash", line_color="black", annotation_text=f"Grand Mean μ = {grand_mu:.1f}")
    
    # Plot connections (shrinkage lines)
    for j in range(n_groups):
        fig.add_trace(go.Scatter(
            x=[j+1, j+1], y=[observed_means[j], partial_pooled[j]],
            mode='lines', line=dict(color='gray', dash='dot', width=1.5),
            showlegend=False
        ))
        
    # Plot No-Pooling (Observed Means)
    fig.add_trace(go.Scatter(
        x=list(range(1, n_groups+1)), y=observed_means,
        mode='markers', name='No Pooling (Raw Means ȳ_j)',
        marker=dict(size=12, color='#1f77b4', symbol='circle')
    ))
    
    # Plot Partial Pooling
    fig.add_trace(go.Scatter(
        x=list(range(1, n_groups+1)), y=partial_pooled,
        mode='markers', name='Partial Pooling (Shrunk θ̂_j)',
        marker=dict(size=14, color='#d62728', symbol='diamond')
    ))
    
    fig.update_layout(
        title="<b>Adaptive Shrinkage Across Groups</b>",
        xaxis=dict(title="Group Identifier $j$", tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(title="Estimated Group Mean"),
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📋 Group Estimates Summary")
    df_summary = pd.DataFrame({
        "Group": [f"Group {j+1}" for j in range(n_groups)],
        "Sample Size (n)": sample_sizes,
        "Raw Mean": np.round(observed_means, 2),
        "Shrunk Mean": np.round(partial_pooled, 2),
        "Shrinkage %": [f"{s:.1f}%" for s in shrinkage_pct]
    })
    st.dataframe(df_summary, use_container_width=True, height=450)

st.markdown("---")
st.markdown(r"""
### 🧠 Key Multilevel Insights
1. **Sample Size Discrepancy:** Notice how groups with small $n$ (e.g. $n=3$) experience large shrinkage towards the grand mean $\mu$, preventing extreme noise from being misinterpreted as real group differences.
2. **Signal-to-Noise Ratio:** If you increase between-group variation $\tau$, groups are shrunk less because real differences between groups are larger.
3. **Avoid Overconfidence:** Partial pooling protects against the twin traps of *overfitting* (no pooling) and *oversimplifying* (complete pooling).
""")
