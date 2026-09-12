"""Build the Monty Hall figure with generated icons embedded in its HTML."""
from pathlib import Path
import base64
from fractions import Fraction

import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT = Path(__file__).resolve().parents[1] / '_static'
ICON_DIR = OUTPUT / 'monty_hall_icons'
OUTPUT.mkdir(exist_ok=True)
ICONS = {
    name: 'data:image/png;base64,' + base64.b64encode((ICON_DIR / f'{name}.png').read_bytes()).decode('ascii')
    for name in ('door', 'goat', 'car')
}
DOORS = [1, 2, 3]
PRIOR = [1 / 3] * 3


def probabilities(pick, opened):
    """Informed host always reveals an unchosen goat; uniform tie-breaking."""
    if pick not in DOORS or opened not in DOORS or pick == opened:
        raise ValueError('The host must open a different door from the player.')
    likelihood = []
    for car in DOORS:
        eligible = [door for door in DOORS if door != pick and door != car]
        likelihood.append(1 / len(eligible) if opened in eligible else 0.0)
    marginal = sum(p * l for p, l in zip(PRIOR, likelihood))
    posterior = [p * l / marginal for p, l in zip(PRIOR, likelihood)]
    return likelihood, marginal, posterior


def fractions(values):
    return [str(Fraction(value).limit_denominator()) for value in values]


def scenario_layout(pick, opened, embedded=False):
    remaining = next(door for door in DOORS if door not in (pick, opened))
    images, annotations, shapes = [], [], []
    for door, center in zip(DOORS, (0.15, 0.5, 0.85)):
        revealed = door == opened
        name = 'goat' if revealed else 'door'
        images.append(dict(
            name=name, source=ICONS[name] if embedded else name,
            xref='paper', yref='paper', x=center, y=1.43,
            sizex=0.21, sizey=0.29, xanchor='center', yanchor='top',
            sizing='contain', layer='above'))
        role = 'YOUR CHOICE · closed' if door == pick else ('HOST REVEALS · goat' if revealed else 'SWITCH OPTION · closed')
        color = '#2563eb' if door == pick else ('#64748b' if revealed else '#15803d')
        annotations.extend([
            dict(x=center, y=1.49, xref='paper', yref='paper', text=f'<b>DOOR {door}</b>',
                 showarrow=False, font=dict(size=16, color='#0f172a'), xanchor='center'),
            dict(x=center, y=1.10, xref='paper', yref='paper', text=role,
                 showarrow=False, font=dict(size=12, color=color), xanchor='center', yanchor='middle'),
        ])
        shapes.append(dict(type='rect', xref='paper', yref='paper',
                           x0=center-0.14, x1=center+0.14, y0=1.06, y1=1.54,
                           fillcolor='#f8fafc', line=dict(color=color, width=1), layer='below'))
    # The prize icon is a legend, never an asserted prize location.
    images.append(dict(name='car', source=ICONS['car'] if embedded else 'car',
                       xref='paper', yref='paper', x=0.97, y=1.72,
                       sizex=0.075, sizey=0.15, xanchor='right', yanchor='top', sizing='contain'))
    annotations.extend([
        dict(x=.89, y=1.66, xref='paper', yref='paper', text='PRIZE TO LOCATE',
             showarrow=False, xanchor='right', font=dict(size=11, color='#475569')),
        dict(x=.5, y=-.22, xref='paper', yref='paper', xanchor='center', showarrow=False,
             text=f'<b>Stay at Door {pick}: 1/3</b> &nbsp; | &nbsp; <b>Switch to Door {remaining}: 2/3</b>'
                  '<br>Closed doors remain uncertain. The car icon is the prize legend.',
             font=dict(size=14, color='#0f172a')),
    ])
    return images, annotations, shapes


def build_figure():
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=('Prior: car location', 'Likelihood: host action', 'Posterior: car location'),
                        horizontal_spacing=.09)
    initial_pick, initial_open = 1, 2
    likelihood, marginal, posterior = probabilities(initial_pick, initial_open)
    for column, (name, values, color) in enumerate([
        ('Prior', PRIOR, '#64748b'), ('Likelihood', likelihood, '#ea580c'),
        ('Posterior', posterior, '#2563eb')], start=1):
        fig.add_trace(go.Bar(
            x=DOORS, y=values, name=name, marker_color=color,
            text=fractions(values), textposition='outside', cliponaxis=False,
            hovertemplate=('Car at Door %{x}<br>' + name + ': %{y:.3f}<extra></extra>')),
            row=1, col=column)
    subplot_annotations = [a.to_plotly_json() for a in fig.layout.annotations]
    images, annotations, shapes = scenario_layout(initial_pick, initial_open, embedded=True)
    scenarios, buttons = [], []
    for pick in DOORS:
        for opened in DOORS:
            if pick == opened:
                continue
            like, marginal, post = probabilities(pick, opened)
            imgs, ann, borders = scenario_layout(pick, opened)
            states = [PRIOR, like, post]
            buttons.append(dict(
                label=f'Pick Door {pick} · host opens {opened}', method='update',
                args=[{'y': states, 'text': [fractions(values) for values in states]},
                      {'images': imgs, 'annotations': subplot_annotations + ann,
                       'shapes': borders}]))
            scenarios.append(dict(pick=pick, opened=opened, likelihood=like,
                                  marginal=marginal, posterior=post))
    fig.update_layout(
        title=dict(text='Monty Hall: what does the host’s choice tell us?',
                   x=.04, y=.98, xanchor='left', yanchor='top', font=dict(size=22)),
        template='plotly_white', showlegend=False, height=820,
        margin=dict(t=310, b=110, l=55, r=30), bargap=.32,
        images=images, annotations=subplot_annotations+annotations, shapes=shapes,
        updatemenus=[dict(buttons=buttons, active=0, x=0, y=1.68,
                         xanchor='left', yanchor='top', bgcolor='white')],
        meta=dict(host_policy='Knows the prize, always reveals an unchosen goat, always offers a switch; uniform tie-breaking.',
                  scenarios=scenarios))
    fig.update_xaxes(tickmode='array', tickvals=DOORS, ticktext=['Door 1', 'Door 2', 'Door 3'],
                     title_text='Hypothesized car location', fixedrange=True)
    fig.update_yaxes(range=[0, 1.16], tickvals=[0, 1/3, 2/3, 1],
                     ticktext=['0', '1/3', '2/3', '1'], fixedrange=True)
    fig.update_yaxes(title_text='Probability', row=1, col=1)
    fig.update_yaxes(title_text='P(opened door | car location)', row=1, col=2)
    return fig


# Reuse the embedded image strings in selector states instead of repeating PNG
# payloads six times. No external image URLs or downloads are needed at runtime.
POST_SCRIPT = """
const gd = document.getElementById('{plot_id}');
const icons = Object.fromEntries(gd.layout.images.map(im => [im.name, im.source]));
for (const button of gd.layout.updatemenus[0].buttons) {
    for (const im of button.args[1].images) im.source = icons[im.name];
}
"""


def main():
    fig = build_figure()
    fig.write_html(OUTPUT / 'monty_hall_demo.html', include_plotlyjs=True,
                   full_html=True, div_id='monty-hall', post_script=POST_SCRIPT,
                   config={'responsive': True, 'displayModeBar': False})
    # Keep three chart panels readable inside narrow book columns; allow scrolling.
    path = OUTPUT / 'monty_hall_demo.html'
    html = path.read_text()
    html = html.replace('<head>', '<head><title>Monty Hall probability explorer</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<style>body{margin:0;background:white;overflow-x:auto;}'
        '#monty-hall{min-width:850px;}</style>', 1)
    path.write_text(html)
    print(f'Built Monty Hall demo with three generated icons and six checked scenarios: {path}')


if __name__ == '__main__':
    main()
