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

    # Centers of the 3 columns (matching horizontal_spacing=0.07 subplots)
    centers = (0.143, 0.500, 0.857)

    for door, center in zip(DOORS, centers):
        revealed = (door == opened)
        name = 'goat' if revealed else 'door'
        images.append(dict(
            name=name, source=ICONS[name] if embedded else name,
            xref='paper', yref='paper', x=center, y=1.63,
            sizex=0.17, sizey=0.31, xanchor='center', yanchor='top',
            sizing='contain', layer='above'))

        if door == pick:
            role = 'YOUR PICK · closed'
            color = '#2563eb'
            border_color = '#2563eb'
            bg_color = '#eff6ff'
        elif revealed:
            role = 'HOST OPENS · goat'
            color = '#dc2626'
            border_color = '#cbd5e1'
            bg_color = '#f8fafc'
        else:
            role = 'SWITCH · closed'
            color = '#16a34a'
            border_color = '#16a34a'
            bg_color = '#f0fdf4'

        annotations.extend([
            dict(x=center, y=1.71, xref='paper', yref='paper', text=f'<b>DOOR {door}</b>',
                 showarrow=False, font=dict(size=13, color='#0f172a'), xanchor='center'),
            dict(x=center, y=1.21, xref='paper', yref='paper', text=f'<b>{role}</b>',
                 showarrow=False, font=dict(size=10.5, color=color), xanchor='center', yanchor='middle'),
        ])
        shapes.append(dict(type='rect', xref='paper', yref='paper',
                           x0=center - 0.125, x1=center + 0.125, y0=1.16, y1=1.76,
                           fillcolor=bg_color, line=dict(color=border_color, width=1.5), layer='below'))

    # Prize icon legend at top right
    images.append(dict(name='car', source=ICONS['car'] if embedded else 'car',
                       xref='paper', yref='paper', x=0.99, y=2.04,
                       sizex=0.06, sizey=0.12, xanchor='right', yanchor='top', sizing='contain'))
    annotations.extend([
        dict(x=0.92, y=1.97, xref='paper', yref='paper',
             text='<b>PRIZE LEGEND</b>',
             showarrow=False, xanchor='right', font=dict(size=10.5, color='#475569')),
        # Clean margin below x-axis for takeaway banner
        dict(x=0.5, y=-0.26, xref='paper', yref='paper', xanchor='center', showarrow=False,
             text=f'<span style="font-size:12.5px;background:#f8fafc;padding:5px 14px;border:1px solid #e2e8f0;border-radius:6px;">'
                  f'<b>Stay at Door {pick}:</b> 1/3 (33.3%) &nbsp;&nbsp;|&nbsp;&nbsp; '
                  f'<b>Switch to Door {remaining}:</b> <span style="color:#16a34a;"><b>2/3 (66.7%)</b></span>'
                  f' &nbsp;&nbsp;—&nbsp;&nbsp; <b>Switching doubles your winning probability!</b></span>',
             font=dict(size=12, color='#0f172a')),
    ])
    return images, annotations, shapes


def build_figure():
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            '<b>Prior</b>: P(Car at Door)',
            '<b>Likelihood</b>: P(Host Opens | Car)',
            '<b>Posterior</b>: P(Car at Door | Action)'
        ),
        horizontal_spacing=0.07
    )

    initial_pick, initial_open = 1, 2
    likelihood, marginal, posterior = probabilities(initial_pick, initial_open)

    for column, (name, values, color) in enumerate([
        ('Prior', PRIOR, '#64748b'),
        ('Likelihood', likelihood, '#ea580c'),
        ('Posterior', posterior, '#2563eb')
    ], start=1):
        fig.add_trace(go.Bar(
            x=DOORS, y=values, name=name, marker_color=color,
            text=fractions(values), textposition='outside', cliponaxis=False,
            textfont=dict(size=12, color='#0f172a', family='system-ui, -apple-system, sans-serif'),
            hovertemplate=('Car at Door %{x}<br>' + name + ': %{y:.3f}<extra></extra>')),
            row=1, col=column)

    # Position subplot titles cleanly right above the plots (y=1.04)
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color='#1e293b', family='system-ui, -apple-system, sans-serif')
        annotation.y = 1.04

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
                label=f'Pick Door {pick} · Host opens {opened}', method='update',
                args=[{'y': states, 'text': [fractions(values) for values in states]},
                      {'images': imgs, 'annotations': subplot_annotations + ann,
                       'shapes': borders}]))
            scenarios.append(dict(pick=pick, opened=opened, likelihood=like,
                                  marginal=marginal, posterior=post))

    fig.update_layout(
        title=dict(
            text='<b>Monty Hall Paradox</b>',
            x=0.01, y=0.98, xanchor='left', yanchor='top',
            font=dict(size=16, color='#0f172a', family='system-ui, -apple-system, sans-serif')
        ),
        template='plotly_white', showlegend=False, height=570,
        margin=dict(t=240, b=85, l=45, r=20), bargap=0.30,
        images=images, annotations=subplot_annotations + annotations, shapes=shapes,
        updatemenus=[dict(
            buttons=buttons, active=0, x=0.22, y=2.04,
            xanchor='left', yanchor='top',
            bgcolor='#ffffff', bordercolor='#cbd5e1', borderwidth=1,
            font=dict(size=11.5, color='#1e293b')
        )],
        meta=dict(host_policy='Knows the prize, always reveals an unchosen goat, always offers a switch; uniform tie-breaking.',
                  scenarios=scenarios))

    fig.update_xaxes(tickmode='array', tickvals=DOORS, ticktext=['Door 1', 'Door 2', 'Door 3'],
                     title_text='<b>Hypothesized car location</b>', title_font=dict(size=11),
                     tickfont=dict(size=11), fixedrange=True)
    fig.update_yaxes(range=[0, 1.20], tickvals=[0, 1/3, 2/3, 1],
                     ticktext=['0', '1/3', '2/3', '1'], fixedrange=True,
                     tickfont=dict(size=10.5))
    fig.update_yaxes(title_text='<b>Probability</b>', title_font=dict(size=11), row=1, col=1)
    fig.update_yaxes(title_text='<b>P(Host opens | Car)</b>', title_font=dict(size=11), row=1, col=2)
    fig.update_yaxes(title_text='<b>Probability</b>', title_font=dict(size=11), row=1, col=3)
    return fig

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
    path = OUTPUT / 'monty_hall_demo.html'
    html = path.read_text()
    html = html.replace('<head>', '<head><title>Monty Hall probability explorer</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<style>'
        'html,body{margin:0;padding:0;overflow:hidden;width:100%;height:100%;background:white;font-family:system-ui,-apple-system,sans-serif;}'
        '#monty-hall{width:100% !important;height:570px !important;overflow:hidden;}'
        '.js-plotly-plot .plotly .modebar{display:none !important;}'
        '</style>', 1)
    path.write_text(html)
    print(f'Successfully built Monty Hall demo: {path}')

if __name__ == '__main__':
    main()
