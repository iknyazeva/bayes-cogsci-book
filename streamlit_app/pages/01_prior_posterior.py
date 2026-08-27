import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Prior, Likelihood & Posterior", page_icon="🎯", layout="wide")

st.title("🎯 Module 1: Prior $\\to$ Likelihood $\\to$ Posterior Explorer")
st.caption("📖 **Coursebook Reference:** [Sessions 2–4: Probability, Likelihood, Priors & Posterior Uncertainty](https://iknyazeva.github.io/bayes-cogsci-book/)")

st.markdown("""
*Context: Synthetic teaching data generated for pedagogical illustration of binary outcomes (e.g. opinion poll support, test accuracy, or policy uptake).*  
Explore how prior beliefs are updated in light of observed binary evidence using the conjugate **Beta-Binomial** model.
""")

# Research Question & Context
with st.expander("📖 Substantive Context & Model Formulation", expanded=False):
    st.markdown(r"""
    **Estimand:** The true underlying proportion $\theta \in [0, 1]$ in a target population.
    
    **Generative Model:**
    $$\theta \sim \text{Beta}(\alpha_{\text{prior}}, \beta_{\text{prior}})$$
    $$k \mid N, \theta \sim \text{Binomial}(N, \theta)$$
    
    **Exact Conjugate Posterior:**
    $$\theta \mid k, N \sim \text{Beta}(\alpha_{\text{prior}} + k, \beta_{\text{prior}} + N - k)$$
    """)

# Sidebar Controls
st.sidebar.header("⚙️ Prior & Data Controls")

if st.sidebar.button("🔄 Reset to Defaults"):
    st.session_state.alpha_prior = 2.0
    st.session_state.beta_prior = 2.0
    st.session_state.n_trials = 30
    st.session_state.k_success = 18
    st.session_state.threshold = 0.5

preset = st.sidebar.selectbox(
    "Prior Preset",
    ["Custom", "Flat / Uniform (Beta(1, 1))", "Weakly Informative (Beta(2, 2))", "Skeptical Centered (Beta(10, 10))", "Optimistic Prior (Beta(15, 5))"]
)

if preset == "Flat / Uniform (Beta(1, 1))":
    default_a, default_b = 1.0, 1.0
elif preset == "Weakly Informative (Beta(2, 2))":
    default_a, default_b = 2.0, 2.0
elif preset == "Skeptical Centered (Beta(10, 10))":
    default_a, default_b = 10.0, 10.0
elif preset == "Optimistic Prior (Beta(15, 5))":
    default_a, default_b = 15.0, 5.0
else:
    default_a = st.session_state.get('alpha_prior', 2.0)
    default_b = st.session_state.get('beta_prior', 2.0)

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    alpha_prior = st.number_input(r"Prior $\alpha$", min_value=0.1, max_value=100.0, value=default_a, step=0.5)
with col_p2:
    beta_prior = st.number_input(r"Prior $\beta$", min_value=0.1, max_value=100.0, value=default_b, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Observed Sample Data")
n_trials = st.sidebar.slider("Sample Size ($N$)", min_value=1, max_value=200, value=st.session_state.get('n_trials', 30), step=1)
k_success = st.sidebar.slider("Observed Successes ($k$)", min_value=0, max_value=n_trials, value=min(st.session_state.get('k_success', 18), n_trials), step=1)

threshold = st.sidebar.slider("Threshold $p_0$ for $P(\\theta > p_0 \\mid \\text{data})$", min_value=0.0, max_value=1.0, value=st.session_state.get('threshold', 0.5), step=0.05)

# Posterior parameters
alpha_post = alpha_prior + k_success
beta_post = beta_prior + n_trials - k_success

# Summary stats
prior_mean = alpha_prior / (alpha_prior + beta_prior)
data_prop = k_success / n_trials if n_trials > 0 else 0
post_mean = alpha_post / (alpha_post + beta_post)
post_mode = (alpha_post - 1) / (alpha_post + beta_post - 2) if (alpha_post > 1 and beta_post > 1) else post_mean

# Equal-Tailed 95% Credible Interval (ETI)
eti_lower = stats.beta.ppf(0.025, alpha_post, beta_post)
eti_upper = stats.beta.ppf(0.975, alpha_post, beta_post)

# Numerical 95% Highest Density Interval (HDI) calculation
def compute_beta_hdi(a, b, cred_mass=0.95, n_grid=2000):
    grid = np.linspace(0.0001, 0.9999, n_grid)
    pdf = stats.beta.pdf(grid, a, b)
    # Sort grid points by PDF height in descending order
    sorted_indices = np.argsort(pdf)[::-1]
    sorted_pdf = pdf[sorted_indices]
    dx = grid[1] - grid[0]
    cumulative_mass = np.cumsum(sorted_pdf * dx)
    hdi_indices = sorted_indices[cumulative_mass <= cred_mass]
    hdi_points = grid[hdi_indices]
    if len(hdi_points) > 0:
        return np.min(hdi_points), np.max(hdi_points)
    return eti_lower, eti_upper

hdi_lower, hdi_upper = compute_beta_hdi(alpha_post, beta_post)
prob_above = 1.0 - stats.beta.cdf(threshold, alpha_post, beta_post)

# Metrics bar
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Prior Mean", f"{prior_mean:.3f}")
m2.metric("Sample Rate ($k/N$)", f"{data_prop:.3f}")
m3.metric("Posterior Mean", f"{post_mean:.3f}", delta=f"{post_mean - prior_mean:+.3f}")
m4.metric("95% Equal-Tailed (ETI)", f"[{eti_lower:.3f}, {eti_upper:.3f}]")
m5.metric("95% Highest Density (HDI)", f"[{hdi_lower:.3f}, {hdi_upper:.3f}]")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 Tripartite Distribution Plot", "⏱️ Sequential Updating Simulation", "🔬 Prior Sensitivity Analysis"])

theta_grid = np.linspace(0.001, 0.999, 500)
prior_pdf = stats.beta.pdf(theta_grid, alpha_prior, beta_prior)
post_pdf = stats.beta.pdf(theta_grid, alpha_post, beta_post)

# Normalized likelihood curve for visual comparison
like_raw = stats.binom.pmf(k_success, n_trials, theta_grid)
dx = theta_grid[1] - theta_grid[0]
like_area = np.sum(like_raw * dx)
like_norm = like_raw / like_area if like_area > 0 else like_raw

with tab1:
    fig = go.Figure()
    
    # Prior curve
    fig.add_trace(go.Scatter(
        x=theta_grid, y=prior_pdf,
        mode='lines', name=f'Prior: Beta({alpha_prior:.1f}, {beta_prior:.1f})',
        line=dict(color='#1f77b4', width=2.5, dash='dash')
    ))
    
    # Normalized Likelihood
    fig.add_trace(go.Scatter(
        x=theta_grid, y=like_norm,
        mode='lines', name=f'Normalized Likelihood (k={k_success}, N={n_trials})',
        line=dict(color='#2ca02c', width=2, dash='dot')
    ))
    
    # Posterior curve
    fig.add_trace(go.Scatter(
        x=theta_grid, y=post_pdf,
        mode='lines', name=f'Posterior: Beta({alpha_post:.1f}, {beta_post:.1f})',
        line=dict(color='#d62728', width=3.5)
    ))
    
    # 95% HDI shaded region
    idx_hdi = (theta_grid >= hdi_lower) & (theta_grid <= hdi_upper)
    fig.add_trace(go.Scatter(
        x=theta_grid[idx_hdi], y=post_pdf[idx_hdi],
        mode='lines', name=f'95% HDI [{hdi_lower:.3f}, {hdi_upper:.3f}]',
        fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.3)',
        line=dict(width=0),
        showlegend=True
    ))

    # Threshold vertical line
    fig.add_vline(x=threshold, line_width=2, line_dash="dash", line_color="black", annotation_text=f"Threshold p0={threshold:.2f} (P(θ > p0)={prob_above:.1%})")

    fig.update_layout(
        title=r"<b>Prior $\to$ Likelihood $\to$ Posterior Redistribution</b>",
        xaxis_title=r"Parameter $\theta$ (Success Probability)",
        yaxis_title="Probability Density",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.8)")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"💡 **Posterior Inference Summary**: Given prior $\\text{{Beta}}({alpha_prior}, {beta_prior})$ and observing ${k_success}$ successes in ${n_trials}$ observations, the posterior probability $P(\\theta > {threshold}) = {prob_above:.1%}$. (95% HDI: [{hdi_lower:.3f}, {hdi_upper:.3f}]).")

with tab2:
    st.subheader("⏱️ Step-by-Step Bayesian Updating as Data Arrives")
    st.markdown("*Synthetic demonstration: Observe how each individual sequential response updates the posterior density.*")
    
    sim_seed = st.number_input("Simulation Random Seed", value=42, step=1)
    np.random.seed(sim_seed)
    
    sim_data = np.random.binomial(1, data_prop if data_prop > 0 else 0.5, size=min(n_trials, 50))
    step = st.slider("Observation Step ($i$)", min_value=0, max_value=len(sim_data), value=min(10, len(sim_data)))
    
    curr_data = sim_data[:step]
    curr_k = int(np.sum(curr_data))
    curr_n = len(curr_data)
    
    curr_a = alpha_prior + curr_k
    curr_b = beta_prior + curr_n - curr_k
    curr_pdf = stats.beta.pdf(theta_grid, curr_a, curr_b)
    
    fig_seq = go.Figure()
    fig_seq.add_trace(go.Scatter(x=theta_grid, y=prior_pdf, mode='lines', name='Initial Prior', line=dict(color='gray', dash='dash')))
    fig_seq.add_trace(go.Scatter(x=theta_grid, y=curr_pdf, mode='lines', name=f'Updated Posterior (Step {step}: {curr_k}/{curr_n})', line=dict(color='#d62728', width=3), fill='tozeroy', fillcolor='rgba(214,39,40,0.2)'))
    
    fig_seq.update_layout(
        title=f"<b>Sequential State at Step {step}/{len(sim_data)}</b> (Observed: {curr_k} successes, {curr_n - curr_k} failures)",
        xaxis_title=r"$\theta$", yaxis_title="Density", template="plotly_white"
    )
    st.plotly_chart(fig_seq, use_container_width=True)

with tab3:
    st.subheader("🔬 Prior Sensitivity Analysis")
    st.markdown("Compare posterior conclusions under 3 distinct prior perspectives with the *same* observed sample data ($k=" + str(k_success) + ", N=" + str(n_trials) + "$):")
    
    priors_comp = {
        "Flat Uniform Beta(1, 1)": (1.0, 1.0),
        "Weakly Informative Beta(2, 2)": (2.0, 2.0),
        "Skeptical Pessimistic Beta(1, 9)": (1.0, 9.0),
        "Skeptical Centered Beta(10, 10)": (10.0, 10.0),
        "Strong Optimistic Beta(18, 2)": (18.0, 2.0),
    }
    
    rows = []
    fig_sens = go.Figure()
    
    for name, (a, b) in priors_comp.items():
        a_p = a + k_success
        b_p = b + n_trials - k_success
        p_mean = a_p / (a_p + b_p)
        h_lo, h_hi = compute_beta_hdi(a_p, b_p)
        p_above = 1.0 - stats.beta.cdf(threshold, a_p, b_p)
        rows.append({
            "Prior": name,
            "Prior Mean": f"{a/(a+b):.3f}",
            "Posterior Mean": f"{p_mean:.3f}",
            "95% HDI": f"[{h_lo:.3f}, {h_hi:.3f}]",
            f"P(θ > {threshold})": f"{p_above:.1%}"
        })
        
        pdf_sens = stats.beta.pdf(theta_grid, a_p, b_p)
        fig_sens.add_trace(go.Scatter(x=theta_grid, y=pdf_sens, mode='lines', name=name))
        
    fig_sens.update_layout(title="<b>Posterior Sensitivity Overlay</b>", xaxis_title=r"$\theta$", yaxis_title="Posterior Density", template="plotly_white")
    st.plotly_chart(fig_sens, use_container_width=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")
st.markdown("""
### 🧠 Guided Inquiry Questions for Students
1. **Prior Influence vs. Sample Size:** When sample size $N$ is small ($N=5$), how much does changing the prior shift the posterior mean? What happens when $N=150$?
2. **HDI vs. ETI:** When is the Equal-Tailed Interval (ETI) nearly identical to the Highest Density Interval (HDI), and when do they diverge (e.g. for skewed or bounded priors)?
3. **Substantive Sensitivity:** In a policy context where approval requires $P(\\theta > 0.5) > 90\\%$, how does a skeptical centered prior change the conclusion compared to a uniform flat prior?
""")
