from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT = Path(__file__).resolve().parents[1] / '_static'
OUTPUT.mkdir(exist_ok=True)

# Monty Hall Bayesian Update
# Doors: 1, 2, 3
# Player picks Door 1. Host opens Door 2.

doors = ['Door 1 (Player Pick)', 'Door 2 (Host Opens)', 'Door 3 (Remaining)']
prior = [1/3, 1/3, 1/3]
# Likelihood of Host opening Door 2, given the prize location
likelihood = [1/2, 0, 1] 
# Posterior = Prior * Likelihood / Marginal
marginal = sum([p * l for p, l in zip(prior, likelihood)])
posterior = [(p * l) / marginal for p, l in zip(prior, likelihood)]

fig = make_subplots(rows=1, cols=3, subplot_titles=('Prior Probability', 'Likelihood (Host Opens Door 2)', 'Posterior Probability'), horizontal_spacing=0.05)

fig.add_trace(go.Bar(x=['Door 1', 'Door 2', 'Door 3'], y=prior, marker_color='#64748b', name='Prior'), row=1, col=1)
fig.add_trace(go.Bar(x=['Door 1', 'Door 2', 'Door 3'], y=likelihood, marker_color='#ea580c', name='Likelihood'), row=1, col=2)
fig.add_trace(go.Bar(x=['Door 1', 'Door 2', 'Door 3'], y=posterior, marker_color='#2563eb', name='Posterior'), row=1, col=3)

fig.update_layout(
    title='Bayesian Updating in the Monty Hall Problem',
    template='plotly_white',
    showlegend=False,
    height=400,
    margin=dict(t=80, b=40, l=40, r=20)
)

fig.update_yaxes(range=[0, 1], title_text='Probability', row=1, col=1)
fig.update_yaxes(range=[0, 1], row=1, col=2)
fig.update_yaxes(range=[0, 1], row=1, col=3)

fig.write_html(OUTPUT / 'monty_hall_demo.html', include_plotlyjs=True, full_html=True, config={'responsive': True})
print("Built Monty Hall demo.")
