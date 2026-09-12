"""Build the synthetic interval and CLT illustrations used in Lesson 1."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

OUTPUT = Path(__file__).resolve().parents[1] / '_static'
OUTPUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# Each dropdown setting distinguishes observations from repeated-study means.
n_experiments = 1000
sample_sizes = [30, 300]
populations = {
    'Uniform(0, 1)': (lambda shape: rng.uniform(0, 1, shape), 0.5, np.sqrt(1 / 12)),
    'Exponential(mean=1)': (lambda shape: rng.exponential(1, shape), 1.0, 1.0),
    'Poisson(mean=3)': (lambda shape: rng.poisson(3, shape), 3.0, np.sqrt(3.0)),
}
fig_clt = make_subplots(rows=1, cols=2, subplot_titles=(
    'Individual observations', 'Means from 1,000 simulated studies'))
buttons = []
settings = len(populations) * len(sample_sizes)
for dist_name, (draw, population_mean, population_sd) in populations.items():
    individual = draw(10000)
    for n in sample_sizes:
        means = draw((n_experiments, n)).mean(axis=1)
        se = population_sd / np.sqrt(n)
        x = np.linspace(min(means.min(), population_mean - 4 * se),
                        max(means.max(), population_mean + 4 * se), 400)
        active = len(buttons) == 0
        fig_clt.add_trace(go.Histogram(
            x=individual, histnorm='probability density', nbinsx=50,
            marker_color='#64748b', name='Individual observations',
            visible=active, showlegend=False), row=1, col=1)
        fig_clt.add_trace(go.Histogram(
            x=means, histnorm='probability density', nbinsx=40,
            marker_color='#2563eb', opacity=0.65, name='Sample means',
            visible=active), row=1, col=2)
        fig_clt.add_trace(go.Scatter(
            x=x, y=norm.pdf(x, loc=population_mean, scale=se),
            line=dict(color='#ea580c', width=3), name='Normal approximation',
            visible=active), row=1, col=2)
        visibility = [False] * (settings * 3)
        first = len(buttons) * 3
        visibility[first:first + 3] = [True] * 3
        buttons.append(dict(
            label=f'{dist_name}; n={n}', method='update',
            args=[{'visible': visibility},
                  {'xaxis.autorange': True, 'xaxis2.autorange': True,
                   'yaxis.autorange': True, 'yaxis2.autorange': True}]))
fig_clt.update_layout(
    title=dict(text='What becomes approximately Normal?', y=0.98, yanchor='top'), template='plotly_white',
    updatemenus=[dict(buttons=buttons, active=0, x=0, y=1.18, xanchor='left')],
    legend=dict(orientation='h', y=-0.2),
    margin=dict(t=145, b=100, l=55, r=30), height=550,
    meta=dict(seed=42, experiments=n_experiments, sample_sizes=sample_sizes))
fig_clt.update_xaxes(title_text='Individual value', row=1, col=1)
fig_clt.update_xaxes(title_text='Mean within one study', row=1, col=2)
fig_clt.update_yaxes(title_text='Density')
fig_clt.write_html(OUTPUT / 'clt_demo.html', include_plotlyjs=True,
                   full_html=True, div_id='lesson1-clt', config={'responsive': True})

# Same observation model and selected dataset for the two interpretations.
rng_intervals = np.random.default_rng(2026)
true_mu, sigma, n, repeats = 5.0, 2.0, 25, 40
means = rng_intervals.normal(true_mu, sigma, (repeats, n)).mean(axis=1)
se = sigma / np.sqrt(n)
z = norm.ppf(0.975)
lo, hi = means - z * se, means + z * se
covers = (lo <= true_mu) & (true_mu <= hi)
selected = 0
prior_mu, prior_sd = 4.0, 1.0
post_var = 1 / (1 / prior_sd**2 + n / sigma**2)
post_mu = post_var * (prior_mu / prior_sd**2 + n * means[selected] / sigma**2)
post_sd = np.sqrt(post_var)
post_lo, post_hi = norm.ppf([0.025, 0.975], loc=post_mu, scale=post_sd)
x = np.linspace(min(prior_mu - 4 * prior_sd, post_mu - 4 * post_sd),
                max(prior_mu + 4 * prior_sd, post_mu + 4 * post_sd), 600)
fig_bf = make_subplots(rows=1, cols=2, subplot_titles=(
    '40 repeated studies: 95% confidence intervals',
    'Highlighted study: prior and posterior'))
for j in range(repeats):
    color = '#2563eb' if j == selected else ('#94a3b8' if covers[j] else '#dc2626')
    fig_bf.add_trace(go.Scatter(
        x=[float(means[j])], y=[j + 1], mode='markers',
        error_x=dict(type='data', array=[z * se], color=color),
        marker=dict(color=color, size=9 if j == selected else 5),
        name=f'Study {j + 1}', showlegend=False,
        hovertemplate=f'Study {j + 1}<br>Mean: %{{x:.2f}}<br>95% CI: [{lo[j]:.2f}, {hi[j]:.2f}]<extra></extra>'
    ), row=1, col=1)
fig_bf.add_vline(x=true_mu, line_dash='dash', line_color='#334155', row=1, col=1)
fig_bf.add_trace(go.Scatter(
    x=x, y=norm.pdf(x, prior_mu, prior_sd), name='Prior',
    line=dict(color='#64748b', dash='dot')), row=1, col=2)
inside = np.linspace(post_lo, post_hi, 200)
fig_bf.add_trace(go.Scatter(
    x=inside, y=norm.pdf(inside, post_mu, post_sd), fill='tozeroy',
    fillcolor='rgba(234, 88, 12, 0.2)', line=dict(width=0),
    name='95% posterior interval'), row=1, col=2)
fig_bf.add_trace(go.Scatter(
    x=x, y=norm.pdf(x, post_mu, post_sd), name='Posterior',
    line=dict(color='#ea580c', width=3)), row=1, col=2)
fig_bf.update_layout(
    title=dict(text=(f'Synthetic Normal observations: n={n}, known SD={sigma:g}<br>'
                     f'<sup>Generating mean={true_mu:g}; {covers.sum()}/{repeats} intervals cover it in this run. '
                     'Blue interval supplies the data for the posterior.</sup>'), font=dict(size=17)),
    template='plotly_white', height=590,
    legend=dict(orientation='h', y=-0.25),
    margin=dict(t=110, b=115, l=60, r=30),
    meta=dict(seed=2026, n=n, observation_sd=sigma, generating_mean=true_mu,
              prior_mean=prior_mu, prior_sd=prior_sd, selected_mean=float(means[selected]),
              posterior_mean=float(post_mu), posterior_sd=float(post_sd)))
fig_bf.update_xaxes(title_text='Estimate and confidence interval', row=1, col=1)
fig_bf.update_yaxes(title_text='Simulated study', autorange='reversed', row=1, col=1)
fig_bf.update_xaxes(title_text='Unknown population mean', row=1, col=2)
fig_bf.update_yaxes(title_text='Density', row=1, col=2)
fig_bf.add_annotation(
    text=f'Prior: Normal({prior_mu:g}, SD={prior_sd:g}); selected sample mean={means[selected]:.2f}',
    xref='paper', yref='paper', x=1, y=-0.17, showarrow=False, xanchor='right')
fig_bf.write_html(OUTPUT / 'bayes_freq_demo.html', include_plotlyjs=True,
                  full_html=True, div_id='lesson1-intervals', config={'responsive': True})
print(f'Built Lesson 1 demos. Coverage: {covers.sum()}/{repeats}; '
      f'posterior: Normal({post_mu:.4f}, SD={post_sd:.4f}).')
