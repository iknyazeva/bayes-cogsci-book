import streamlit as st

st.set_page_config(
    page_title="Bayesian Analysis of Empirical Data — Interactive Demos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Bayesian Analysis of Empirical Data")
st.subheader("Interactive Demonstration Platform for Empirical & Cognitive Sciences")

st.markdown(r"""
Welcome to the interactive demonstration companion for the course **Bayesian Analysis of Empirical Data**.

This platform provides lightweight, real-time visual tools to build mathematical intuition for core Bayesian concepts before running heavy Markov Chain Monte Carlo models in **PyMC 5**.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(r"""
    ### 🎯 Module 1: Prior $\to$ Likelihood $\to$ Posterior
    Explore how prior distributions combine with observed data through the likelihood function.
    - **Beta-Binomial Conjugate Updating**
    - **95% Credible Intervals vs. HDI**
    - **Sequential Data Arrival Simulation**
    - **Prior Sensitivity Analysis**
    
    👉 Navigate to **** in the sidebar.
    """)

with col2:
    st.success(r"""
    ### 🔄 Module 2: MCMC & Metropolis-Hastings
    Visualize how Markov chains explore parameter space and why proposal tuning matters.
    - **Proposal Width ($\sigma_{\text{prop}}$) Tuning**
    - **Step-by-Step Accept/Reject Decisions**
    - **Traceplots, Histograms, and Autocorrelation**
    - **ESS & $\hat{R}$ Diagnostics**
    
    👉 Navigate to **** in the sidebar.
    """)

with col3:
    st.warning(r"""
    ### 🌲 Module 3: Partial Pooling & Shrinkage
    Understand how multilevel and hierarchical models share information across groups.
    - **No Pooling vs. Complete Pooling vs. Partial Pooling**
    - **Shrinkage as a function of group sample size ($)**
    - **Between-group vs. Within-group variation**
    
    👉 Navigate to **** in the sidebar.
    """)

st.divider()

st.markdown(r"""
### 📚 Course Resources & Navigation
- 📖 **Online Textbook**: [Jupyter Book on GitHub Pages](https://iknyazeva.github.io/bayes-cogsci-book/)
- 💻 **Source Code & Notebooks**: [GitHub Repository](https://github.com/iknyazeva/bayes-cogsci-book)
- 🚀 **Google Colab**: Each chapter in the book includes an *"Open in Colab"* launch badge for running PyMC 5 models with free GPU/CPU compute.
""")
