import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Partial Pooling & Shrinkage", page_icon="🌲", layout="wide")

st.title("🌲 Module 3: Multilevel Models & Partial Pooling Shrinkage")
st.caption("📖 **Coursebook Reference:** [Session 10: Hierarchical Models & Partial Pooling](https://iknyazeva.github.io/bayes-cogsci-book/)")

st.markdown("""
*Context: Synthetic teaching data generated for pedagogical illustration of multilevel / hierarchical shrinkage.*  
This demonstration illustrates the **analytical mechanics of shrinkage** (Empirical Bayes weight formulation) before fitting full varying-intercept models in **PyMC 5**.
""")

with st.expander("📖 Generative Model & Analytical Shrinkage Formula", expanded=False):
    st.markdown(r"""
    **Two-Level Gaussian Hierarchy:**
    $$\theta_j \sim \mathcal{N}(\mu, \tau^2) \quad \text{(Group Means)}$$
    $$y_{ij} \sim \mathcal{N}(\theta_j, \sigma^2) \quad \text{(Individual Observations within Group } j)$$
    
    **Analytical Shrinkage Weight (Known Parameters Illustration):**
    $$\hat{\theta}_j = w_j \bar{y}_j + (1 - w_j) \mu \quad \text{where} \quad w_j = \frac{\tau^2}{\tau^2 + \sigma^2 / n_j}$$
    - **Small group ($n_j$ small)** $\implies w_j \to 0 \implies \hat{\theta}_j \to \mu$ (strong shrinkage toward population grand mean).
    - **Large group ($n_j$ large)** $\implies w_j \to 1 \implies \hat{\theta}_j \to \bar{y}_j$ (sample data dominates).
    """)

st.sidebar.header("⚙️ Hierarchy Parameters")

if st.sidebar.button("🔄 Reset to Defaults"):
    st.session_state.grand_mu = 50.0
    st.session_state.tau = 8.0
    st.session_state.sigma = 12.0
    st.session_state.n_groups = 8

grand_mu = st.sidebar.slider(r"Population Grand Mean $\mu$", min_value=0.0, max_value=100.0, value=st.session_state.get('grand_mu', 50.0), step=1.0)
tau = st.sidebar.slider(r"Between-Group SD $\tau$", min_value=1.0, max_value=25.0, value=st.session_state.get('tau', 8.0), step=0.5, help="Heterogeneity across groups")
sigma = st.sidebar.slider(r"Within-Group Noise $\sigma$", min_value=1.0, max_value=30.0, value=st.session_state.get('sigma', 12.0), step=0.5, help="Individual measurement variation")

n_groups = st.sidebar.slider("Number of Groups ($J$)", min_value=4, max_value=20, value=st.session_state.get('n_groups', 8), step=1)
seed = st.sidebar.number_input("Random Data Seed", value=42, step=1)
np.random.seed(seed)

sample_sizes = np.random.choice([3, 5, 8, 15, 30, 60], size=n_groups)
true_thetas = np.random.normal(grand_mu, tau, size=n_groups)

observed_means = np.zeros(n_groups)
for j in range(n_groups):
    y_obs = np.random.normal(true_thetas[j], sigma, size=sample_sizes[j])
    observed_means[j] = np.mean(y_obs)

weights = (tau**2) / (tau**2 + (sigma**2) / sample_sizes)
partial_pooled = weights * observed_means + (1.0 - weights) * grand_mu
shrinkage_pct = (1.0 - weights) * 100.0

col1, col2 = st.columns([0.65, 0.35])

with col1:
    fig = go.Figure()
    
    fig.add_hline(y=grand_mu, line_dash="dash", line_color="black", annotation_text=f"Grand Mean μ = {grand_mu:.1f}")
    
    for j in range(n_groups):
        fig.add_trace(go.Scatter(
            x=[j+1, j+1], y=[observed_means[j], partial_pooled[j]],
            mode='lines', line=dict(color='gray', dash='dot', width=1.5),
            showlegend=False
        ))
        
    fig.add_trace(go.Scatter(
        x=list(range(1, n_groups+1)), y=observed_means,
        mode='markers', name='No Pooling (Raw Means ȳ_j)',
        marker=dict(size=12, color='#1f77b4', symbol='circle')
    ))
    
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
st.markdown("""
### 🧠 Guided Inquiry Questions for Students
1. **Sample Size Discrepancy:** Why do groups with $n=3$ shrink drastically toward the grand mean $\mu$, while groups with $n=60$ remain close to their raw mean?
2. **Between-Group Variance Effect:** If $\\tau$ increases (high group heterogeneity), what happens to the shrinkage percentage across all groups?
3. **Overconfidence Protection:** How does partial pooling protect researchers from false discoveries when comparing performance rankings across schools or clinics with unequal sample sizes?
""")
