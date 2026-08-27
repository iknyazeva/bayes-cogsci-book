import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(page_title="MCMC & Metropolis-Hastings", page_icon="🔄", layout="wide")

st.title("🔄 Module 2: Interactive Metropolis-Hastings MCMC Sampler")
st.markdown(r"""
*Course: Bayesian Analysis of Empirical Data (Chapter 6 & 11)*  
Understand the mechanics of Markov Chain Monte Carlo (MCMC) sampling, proposal tuning, and convergence diagnostics before running modern samplers in **PyMC 5**.
""")

with st.expander("📖 Target Distributions & Algorithm Details", expanded=False):
    st.markdown(r"""
    **Metropolis-Hastings Step Rule (Symmetric Proposal):**
    1. At step $t$, current state is $\theta^{(t)}$. Propose candidate $\theta^* \sim \mathcal{N}(\theta^{(t)}, \sigma_{\text{prop}}^2)$.
    2. Compute acceptance probability:
       $$\alpha(\theta^{(t)}, \theta^*) = \min\left(1, \frac{p(\theta^*)}{p(\theta^{(t)})}\right)$$
    3. Draw $u \sim \text{Uniform}(0, 1)$. If $u \le \alpha$, accept: $\theta^{(t+1)} = \theta^*$; else reject: $\theta^{(t+1)} = \theta^{(t)}$.
    """)

# Sidebar Controls
st.sidebar.header("⚙️ Sampler Settings")

target_choice = st.sidebar.selectbox(
    "Target Distribution $p(\\theta)$",
    ["Standard Normal N(0, 1)", "Bimodal Gaussian Mixture (Modes at -2 & +2)", "Skewed / Heavy-Tailed Student-t (df=3)"]
)

def target_pdf(x):
    if target_choice == "Standard Normal N(0, 1)":
        return stats.norm.pdf(x, 0, 1)
    elif target_choice == "Bimodal Gaussian Mixture (Modes at -2 & +2)":
        return 0.55 * stats.norm.pdf(x, -2.0, 0.8) + 0.45 * stats.norm.pdf(x, 2.0, 0.6)
    else:
        return stats.t.pdf(x, df=3, loc=0, scale=1)

sigma_prop = st.sidebar.slider(
    r"Proposal Standard Deviation $\sigma_{\text{prop}}$",
    min_value=0.05, max_value=8.0, value=1.5, step=0.05,
    help="Tuning width: Very small = stuck local walk (high acceptance); Very large = frequent rejection; ~1.5-2.4 = optimal mixing."
)

if sigma_prop < 0.2:
    st.sidebar.warning("⚠️ **Proposal too narrow**: High acceptance rate, but very slow exploration (high autocorrelation)!")
elif sigma_prop > 4.5:
    st.sidebar.error("⚠️ **Proposal too wide**: Chain frequently proposes low-density points and gets stuck (low acceptance rate)!")
else:
    st.sidebar.success("✅ **Proposal well-tuned**: Balances exploration and acceptance (~20–45%).")

tab_sim, tab_step, tab_diag = st.tabs(["🚀 Full Simulation & Traceplots", "👣 Step-by-Step Interactive Walk", r"📊 Diagnostics ($\hat{R}$ & ESS)"])

# Simulation tab
with tab_sim:
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        n_samples = st.slider("Total Draws ($N$)", min_value=200, max_value=5000, value=1500, step=100)
    with col_s2:
        n_burnin = st.slider("Warmup / Burn-in", min_value=0, max_value=1000, value=200, step=50)
    with col_s3:
        n_chains = st.slider("Number of Parallel Chains", min_value=1, max_value=4, value=2, step=1)
        
    run_seed = st.number_input("Random Seed", value=123, step=1)
    np.random.seed(run_seed)
    
    # Run chains
    chains = []
    acceptances = []
    
    for c in range(n_chains):
        # Stagger initial values across chains
        init_val = np.random.uniform(-4, 4) if c > 0 else 0.0
        chain = np.zeros(n_samples)
        chain[0] = init_val
        n_acc = 0
        
        for t in range(1, n_samples):
            current = chain[t-1]
            proposal = np.random.normal(current, sigma_prop)
            
            p_curr = target_pdf(current)
            p_prop = target_pdf(proposal)
            
            alpha = min(1.0, p_prop / (p_curr + 1e-12))
            if np.random.uniform(0, 1) <= alpha:
                chain[t] = proposal
                n_acc += 1
            else:
                chain[t] = current
                
        chains.append(chain)
        acceptances.append(n_acc / (n_samples - 1))
        
    avg_acc = np.mean(acceptances)
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average Acceptance Rate", f"{avg_acc:.1%}", delta="Optimal ~23-45%")
    m2.metric("Retained Samples / Chain", f"{n_samples - n_burnin}")
    
    # Retained samples across all chains
    retained_chains = [c[n_burnin:] for c in chains]
    all_retained = np.concatenate(retained_chains)
    m3.metric("Posterior Mean", f"{np.mean(all_retained):.3f}")
    m4.metric("Posterior SD", f"{np.std(all_retained):.3f}")
    
    st.markdown("---")
    
    # Plotly visualization: Trace plot & Histogram
    fig_mcmc = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=("Markov Chain Trace Plots", "Sampled Histogram vs. True Target Density")
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, c in enumerate(chains):
        fig_mcmc.add_trace(
            go.Scatter(y=c, mode='lines', name=f'Chain {idx+1} (Acc: {acceptances[idx]:.1%})', line=dict(color=colors[idx % len(colors)], width=1.2)),
            row=1, col=1
        )
    
    # Add burnin vertical line
    fig_mcmc.add_vline(x=n_burnin, line_width=1.5, line_dash="dash", line_color="black", row=1, col=1)
    
    # Histogram of retained
    fig_mcmc.add_trace(
        go.Histogram(x=all_retained, histnorm='probability density', nbinsx=50, name='Sampled Draws', marker_color='rgba(31, 119, 180, 0.5)'),
        row=1, col=2
    )
    
    # True target density overlay
    x_grid = np.linspace(-6, 6, 400)
    fig_mcmc.add_trace(
        go.Scatter(x=x_grid, y=target_pdf(x_grid), mode='lines', name='True Target p(θ)', line=dict(color='black', width=2.5, dash='dash')),
        row=1, col=2
    )
    
    fig_mcmc.update_layout(template="plotly_white", height=450, hovermode="x unified")
    fig_mcmc.update_xaxes(title_text="Iteration (Draw)", row=1, col=1)
    fig_mcmc.update_yaxes(title_text=r"Parameter $\theta$", row=1, col=1)
    fig_mcmc.update_xaxes(title_text=r"Parameter $\theta$", row=1, col=2)
    fig_mcmc.update_yaxes(title_text="Density", row=1, col=2)
    
    st.plotly_chart(fig_mcmc, use_container_width=True)

# Step-by-step tab
with tab_step:
    st.subheader("👣 Single-Step Metropolis-Hastings Visualizer")
    st.markdown("Inspect exactly what happens inside a single MCMC step: proposal generation, ratio calculation, and probabilistic acceptance.")
    
    col_step1, col_step2 = st.columns(2)
    with col_step1:
        curr_theta = st.slider("Current State $\\theta^{(t)}$", min_value=-4.0, max_value=4.0, value=0.0, step=0.1)
    with col_step2:
        prop_seed = st.number_input("Step Random Draw Seed", value=7, step=1)
        
    np.random.seed(prop_seed)
    cand_theta = np.random.normal(curr_theta, sigma_prop)
    u_draw = np.random.uniform(0, 1)
    
    p_c = target_pdf(curr_theta)
    p_p = target_pdf(cand_theta)
    alpha_ratio = min(1.0, p_p / (p_c + 1e-12))
    accepted = (u_draw <= alpha_ratio)
    
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("Proposed Candidate $\\theta^*$", f"{cand_theta:.3f}")
    res_col2.metric("Acceptance Ratio $\\alpha$", f"{alpha_ratio:.3f}")
    res_col3.metric("Uniform Draw $u$", f"{u_draw:.3f}")
    if accepted:
        res_col4.success("✅ **ACCEPTED** ($\theta^{(t+1)} = \theta^*$)")
    else:
        res_col4.error("❌ **REJECTED** ($\theta^{(t+1)} = \theta^{(t)}$)")
        
    # Step plot
    x_plot = np.linspace(-6, 6, 400)
    fig_step = go.Figure()
    fig_step.add_trace(go.Scatter(x=x_plot, y=target_pdf(x_plot), mode='lines', name='Target p(θ)', line=dict(color='black', width=2)))
    
    # Proposal distribution around current state
    prop_pdf = stats.norm.pdf(x_plot, curr_theta, sigma_prop)
    fig_step.add_trace(go.Scatter(x=x_plot, y=prop_pdf, mode='lines', name=f'Proposal N({curr_theta:.1f}, {sigma_prop:.2f}²)', line=dict(color='gray', dash='dot')))
    
    # Points
    fig_step.add_trace(go.Scatter(x=[curr_theta], y=[p_c], mode='markers', marker=dict(size=14, color='blue'), name='Current State θ(t)'))
    fig_step.add_trace(go.Scatter(x=[cand_theta], y=[p_p], mode='markers', marker=dict(size=16, color='green' if accepted else 'red', symbol='star'), name='Candidate θ*'))
    
    fig_step.update_layout(title="<b>Single Step Proposal and Evaluation</b>", xaxis_title=r"$\theta$", yaxis_title="Density", template="plotly_white")
    st.plotly_chart(fig_step, use_container_width=True)

# Diagnostics tab
with tab_diag:
    st.subheader(r"📊 MCMC Diagnostics: Autocorrelation, ESS, and $\hat{R}$")
    
    # Autocorrelation
    def autocorr(x, max_lag=40):
        n = len(x)
        x_mean = np.mean(x)
        var = np.var(x)
        if var == 0:
            return np.ones(max_lag)
        autocov = np.correlate(x - x_mean, x - x_mean, mode='full')[n - 1:n + max_lag] / n
        return autocov / var
        
    lags = 40
    acf_chain1 = autocorr(retained_chains[0], max_lag=lags)
    
    # Simple ESS approximation
    act = 1.0 + 2.0 * np.sum(acf_chain1[1:min(15, len(acf_chain1))])
    ess_est = len(retained_chains[0]) / max(1.0, act)
    
    # Simple Gelman-Rubin R-hat across chains
    if len(retained_chains) > 1:
        M = len(retained_chains)
        N = len(retained_chains[0])
        chain_means = [np.mean(c) for c in retained_chains]
        grand_mean = np.mean(chain_means)
        B = N / (M - 1) * np.sum((chain_means - grand_mean)**2)
        W = np.mean([np.var(c, ddof=1) for c in retained_chains])
        var_plus = (N - 1) / N * W + (1 / N) * B
        r_hat = np.sqrt(var_plus / (W + 1e-12))
    else:
        r_hat = 1.0
        
    d1, d2, d3 = st.columns(3)
    d1.metric("Effective Sample Size (ESS)", f"{int(ess_est)}", help="Number of independent draws equivalent to correlated MCMC sample.")
    d2.metric(r"Gelman-Rubin $\hat{R}$", f"{r_hat:.3f}", delta="Target: < 1.01")
    d3.metric("Autocorrelation at Lag 1", f"{acf_chain1[1]:.3f}")
    
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=list(range(lags)), y=acf_chain1, name='Autocorrelation', marker_color='#1f77b4'))
    fig_acf.update_layout(title="<b>Autocorrelation Function (Chain 1)</b>", xaxis_title="Lag", yaxis_title="Autocorrelation", template="plotly_white")
    st.plotly_chart(fig_acf, use_container_width=True)

st.markdown("---")
st.markdown(r"""
### 🧠 Guided Inquiry Questions for Students
1. **Proposal Tuning:** What happens to the acceptance rate and traceplot when you set $\\sigma_{\\text{prop}} = 0.05$? Why is a 95% acceptance rate bad in this case?
2. **Multimodal Exploration:** For the Bimodal target, test $\\sigma_{\\text{prop}} = 0.5$ vs $\\sigma_{\\text{prop}} = 2.5$. Which one gets trapped in a single mode?
3. **Diagnostics Rule:** When can you trust MCMC estimates? (Course gate: $\\hat{R} < 1.01$, $\\text{ESS} > 400$, 0 divergences).
""")
