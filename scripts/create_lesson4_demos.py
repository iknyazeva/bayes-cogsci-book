import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

os.makedirs('_static', exist_ok=True)

# ---------------------------------------------------------
# 1. Credible vs Confidence Intervals
# ---------------------------------------------------------
def create_cred_vs_conf():
    theta_true = 0.55
    N = 100
    n_sims = 100
    k_obs = 55
    x = np.linspace(0.35, 0.75, 500)
    y_post = stats.beta.pdf(x, k_obs + 1, N - k_obs + 1)
    
    np.random.seed(42)
    k_sims = np.random.binomial(N, theta_true, n_sims)
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=["<b>Bayesian Credible Interval</b><br><span style='font-size:12px;color:gray'>95% Probability mass (Data is fixed)</span>",
                                        "<b>Frequentist Confidence Intervals</b><br><span style='font-size:12px;color:gray'>95% Coverage across repeated samples</span>"])
    
    ci_lower = stats.beta.ppf(0.025, k_obs + 1, N - k_obs + 1)
    ci_upper = stats.beta.ppf(0.975, k_obs + 1, N - k_obs + 1)
    x_fill = np.linspace(ci_lower, ci_upper, 100)
    y_fill = stats.beta.pdf(x_fill, k_obs + 1, N - k_obs + 1)
    
    fig.add_trace(go.Scatter(x=x, y=y_post, mode='lines', line=dict(color='#2b6cb0', width=3), name="Posterior Density"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.concatenate([x_fill, x_fill[::-1]]), 
                             y=np.concatenate([y_fill, np.zeros_like(y_fill)]), 
                             fill='toself', fillcolor='rgba(43, 108, 176, 0.4)', line=dict(color='rgba(255,255,255,0)'), 
                             name="95% Credible Interval"), row=1, col=1)
    
    # Text annotation for Credible Interval
    fig.add_annotation(x=0.55, y=2.0, text=f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]", showarrow=False, font=dict(color='#2b6cb0', size=14), row=1, col=1)

    fig.add_vline(x=theta_true, line=dict(color='#e53e3e', width=2, dash='dash'), row=1, col=1)
    
    for i in range(n_sims):
        p_hat = k_sims[i] / N
        se = np.sqrt(p_hat * (1 - p_hat) / N)
        l = p_hat - 1.96 * se
        u = p_hat + 1.96 * se
        covers = (l <= theta_true) and (u >= theta_true)
        color = '#a0aec0' if covers else '#e53e3e'
        width = 1.5 if covers else 2.5
        
        fig.add_trace(go.Scatter(x=[l, u], y=[i, i], mode='lines', line=dict(color=color, width=width), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[p_hat], y=[i], mode='markers', marker=dict(color=color, size=3), showlegend=False), row=1, col=2)
        
    fig.add_vline(x=theta_true, line=dict(color='#e53e3e', width=2, dash='dash'), row=1, col=2)
    fig.add_annotation(x=theta_true+0.01, y=n_sims-5, text="True θ (Fixed)", showarrow=False, font=dict(color='#e53e3e', size=12), xanchor="left", row=1, col=2)
    
    fig.update_layout(template='plotly_white', height=450, showlegend=False, margin=dict(l=40, r=40, t=80, b=40))
    fig.update_xaxes(title_text="Parameter θ", range=[0.35, 0.75])
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="100 Simulated Samples", showticklabels=False, row=1, col=2)
    
    fig.write_html("_static/lesson4_cred_vs_conf.html", include_plotlyjs='cdn')

# ---------------------------------------------------------
# 2. ETI vs HDI
# ---------------------------------------------------------
def create_eti_hdi():
    a = 2.0
    scale = 1.0
    x = np.linspace(0, 8, 500)
    y = stats.gamma.pdf(x, a, scale=scale)
    
    eti_l = stats.gamma.ppf(0.025, a, scale=scale)
    eti_u = stats.gamma.ppf(0.975, a, scale=scale)
    
    from scipy.optimize import fsolve
    def equations(p):
        L, U = p
        eq1 = stats.gamma.pdf(L, a) - stats.gamma.pdf(U, a)
        eq2 = stats.gamma.cdf(U, a) - stats.gamma.cdf(L, a) - 0.95
        return (eq1, eq2)
    hdi_l, hdi_u = fsolve(equations, (0.1, 5.0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#4a5568', width=3), name="Posterior Density"))
    
    # ETI Bar (Orange)
    fig.add_trace(go.Scatter(x=[eti_l, eti_u], y=[-0.02, -0.02], mode='lines+markers', 
                             line=dict(color='#dd6b20', width=6), marker=dict(size=10, symbol='line-ns'),
                             name="95% Equal-Tailed Interval (ETI)"))
    fig.add_annotation(x=(eti_l+eti_u)/2, y=-0.04, text=f"ETI (Wider, pushed to tail): [{eti_l:.2f}, {eti_u:.2f}]", 
                       showarrow=False, font=dict(color='#dd6b20', size=12))

    # HDI Bar (Teal)
    fig.add_trace(go.Scatter(x=[hdi_l, hdi_u], y=[-0.06, -0.06], mode='lines+markers', 
                             line=dict(color='#319795', width=6), marker=dict(size=10, symbol='line-ns'),
                             name="95% Highest Density Interval (HDI)"))
    fig.add_annotation(x=(hdi_l+hdi_u)/2, y=-0.08, text=f"HDI (Shorter, captures peak): [{hdi_l:.2f}, {hdi_u:.2f}]", 
                       showarrow=False, font=dict(color='#319795', size=12))

    # Shade HDI by default as a subtle background
    x_hdi = np.linspace(hdi_l, hdi_u, 100)
    y_hdi = stats.gamma.pdf(x_hdi, a)
    fig.add_trace(go.Scatter(x=np.concatenate([x_hdi, x_hdi[::-1]]), 
                             y=np.concatenate([y_hdi, np.zeros_like(y_hdi)]), 
                             fill='toself', fillcolor='rgba(49, 151, 149, 0.15)', line=dict(width=0), 
                             showlegend=False))
                             
    # Shade ETI as an even more subtle background
    x_eti = np.linspace(eti_l, eti_u, 100)
    y_eti = stats.gamma.pdf(x_eti, a)
    fig.add_trace(go.Scatter(x=np.concatenate([x_eti, x_eti[::-1]]), 
                             y=np.concatenate([y_eti, np.zeros_like(y_eti)]), 
                             fill='toself', fillcolor='rgba(221, 107, 32, 0.1)', line=dict(width=0), 
                             showlegend=False))

    fig.update_layout(
        title="<b>ETI vs HDI on a Skewed Posterior</b><br><span style='font-size:12px;color:gray'>The HDI shifts left to capture the most probable values. ETI guarantees 2.5% in each tail.</span>",
        template='plotly_white', height=450,
        legend=dict(y=0.9, x=0.55),
        yaxis=dict(range=[-0.12, max(y)+0.05])
    )
    fig.update_xaxes(title_text="Parameter Value")
    fig.update_yaxes(title_text="Density", showticklabels=False)
    
    fig.write_html("_static/lesson4_eti_hdi.html", include_plotlyjs='cdn')

# ---------------------------------------------------------
# 3. Hypothesis ROPE
# ---------------------------------------------------------
def create_rope():
    mu_diff = 0.08
    sigma_diff = 0.03
    x = np.linspace(-0.02, 0.20, 500)
    y = stats.norm.pdf(x, mu_diff, sigma_diff)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#2b6cb0', width=3), name="Posterior Difference"))
    
    t_init = 0.05
    x_fill = np.linspace(t_init, 0.20, 100)
    y_fill = stats.norm.pdf(x_fill, mu_diff, sigma_diff)
    prob = 1 - stats.norm.cdf(t_init, mu_diff, sigma_diff)
    
    fig.add_trace(go.Scatter(x=np.concatenate([x_fill, x_fill[::-1]]), 
                             y=np.concatenate([y_fill, np.zeros_like(y_fill)]), 
                             fill='toself', fillcolor='rgba(43, 108, 176, 0.5)', line=dict(width=0), 
                             name=f"Prob > {t_init}"))
                             
    fig.add_vline(x=t_init, line=dict(color='#e53e3e', width=2, dash='dash'))
    fig.add_annotation(x=t_init, y=max(y)*0.9, text="Threshold (5 points)", font=dict(color='#e53e3e'), xanchor="right", yanchor="bottom", textangle=-90)

    fig.update_layout(
        title=f"<b>Hypothesis Testing: Probability that Difference > 5 points</b><br><span style='font-size:13px;color:gray'>Area under curve = P(Δ > 0.05 | Data) = {prob:.1%}</span>",
        template='plotly_white', height=400,
        xaxis_title="Difference in Support (Group A - Group B)",
        yaxis_title="Posterior Density",
        showlegend=False,
        margin=dict(t=80)
    )
    fig.add_annotation(x=0.10, y=stats.norm.pdf(0.10, mu_diff, sigma_diff)/2, text=f"<b>{prob:.1%}</b>", showarrow=False, font=dict(size=18, color='white'))
                       
    fig.write_html("_static/lesson4_rope.html", include_plotlyjs='cdn')

# ---------------------------------------------------------
# 4. Prediction Separator
# ---------------------------------------------------------
def create_prediction():
    x = np.linspace(15, 85, 500)
    y_mean = stats.norm.pdf(x, 50, 2)
    y_pred = stats.norm.pdf(x, 50, 10.2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=x, y=y_mean, mode='lines', fill='tozeroy', 
                             fillcolor='rgba(49, 151, 149, 0.6)', line=dict(color='#234e52', width=2), 
                             name="<b>Expected Mean (μ)</b><br>Narrow Epistemic Uncertainty"))
                             
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', fill='tozeroy', 
                             fillcolor='rgba(221, 107, 32, 0.3)', line=dict(color='#c05621', width=2), 
                             name="<b>Future Observation (y)</b><br>Wide Aleatory Uncertainty"))

    fig.update_layout(
        title="<b>Parameter Uncertainty vs. Predictive Uncertainty</b>",
        template='plotly_white', height=450,
        xaxis_title="Outcome Scale",
        yaxis_title="Density",
        legend=dict(x=0.02, y=0.98, bordercolor="gray", borderwidth=1, bgcolor="rgba(255,255,255,0.9)")
    )
    
    fig.write_html("_static/lesson4_prediction.html", include_plotlyjs='cdn')

if __name__ == "__main__":
    create_cred_vs_conf()
    create_eti_hdi()
    create_rope()
    create_prediction()
    print("Lesson 4 demos recreated with vastly improved clarity.")
