import streamlit as st

st.set_page_config(
    page_title="Bayesian Analysis of Empirical Data — Interactive Demos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Bayesian Analysis of Empirical Data: Interactive Demonstrations")
st.markdown("""
*Interactive pedagogical companion for the 14-session course.*  
This platform provides lightweight, real-time visual tools to build mathematical intuition before fitting full Bayesian models in **PyMC 5**.
""")

st.info("""
> 💡 **Notice**: These demonstrations are interactive visual tools designed to accompany the course readings. They run client-side simulations and exact analytical updating without requiring external server dependencies or AI tools.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎯 Module 1: Prior $\\to$ Likelihood $\\to$ Posterior
    *Relevant Sessions: 2–4*  
    Explore how prior distributions combine with observed data through the likelihood function.
    - **Beta-Binomial Conjugate Updating**
    - **Equal-Tailed (ETI) vs. Highest Density Intervals (HDI)**
    - **Sequential Data Arrival Simulation**
    - **Prior Sensitivity Analysis**
    
    👉 Navigate to **`01_prior_posterior`** in the sidebar.
    """)

with col2:
    st.markdown("""
    ### 🔄 Module 2: MCMC & Metropolis-Hastings
    *Relevant Sessions: 6 & 11*  
    Visualize how Markov chains explore parameter space and why proposal tuning matters.
    - **Proposal Width ($\\sigma_{\\text{prop}}$) Tuning**
    - **Step-by-Step Accept/Reject Decisions**
    - **Traceplots, Histograms, and Autocorrelation**
    - **Teaching ESS & $\\hat{R}$ Diagnostics**
    
    👉 Navigate to **`02_mcmc_sampler`** in the sidebar.
    """)

with col3:
    st.markdown("""
    ### 🌲 Module 3: Partial Pooling & Shrinkage
    *Relevant Session: 10*  
    Understand how multilevel and hierarchical models share information across groups.
    - **No Pooling vs. Complete Pooling vs. Partial Pooling**
    - **Adaptive shrinkage as a function of group sample size ($n_j$)**
    - **Analytical shrinkage illustration**
    
    👉 Navigate to **`03_partial_pooling`** in the sidebar.
    """)

st.divider()

st.markdown("""
### 📚 Course Resources & Navigation
- 📖 **Online Coursebook**: [Jupyter Book](https://iknyazeva.github.io/bayes-cogsci-book/)
- 💻 **Source Code & Notebooks**: [GitHub Repository `iknyazeva/bayes-cogsci-book`](https://github.com/iknyazeva/bayes-cogsci-book)
- 🚀 **Google Colab**: Computational laboratory notebooks include optional launch links to Google Colab.
""")
