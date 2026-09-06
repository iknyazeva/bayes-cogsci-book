import os
import shutil
import numpy as np
import scipy.stats as stats

OUT_DIR = "_static/dist_icons"
os.makedirs(OUT_DIR, exist_ok=True)

def make_continuous_svg(x, y, color="#2563eb", fill_color="rgba(37,99,235,0.18)", baseline=42, top=6, left=6, right=94, bg_curve=None):
    max_y = np.max(y) if np.max(y) > 0 else 1.0
    if bg_curve is not None:
        max_y = max(max_y, np.max(bg_curve[1]))
        
    scaled_y = baseline - (y / max_y) * (baseline - top)
    scaled_x = left + (x - x[0]) / (x[-1] - x[0]) * (right - left)
    
    fill_pts = [f"{scaled_x[0]:.1f},{baseline:.1f}"]
    for sx, sy in zip(scaled_x, scaled_y):
        fill_pts.append(f"{sx:.1f},{sy:.1f}")
    fill_pts.append(f"{scaled_x[-1]:.1f},{baseline:.1f}")
    fill_d = "M " + " L ".join(fill_pts) + " Z"
    
    stroke_pts = [f"{sx:.1f},{sy:.1f}" for sx, sy in zip(scaled_x, scaled_y)]
    stroke_d = "M " + " L ".join(stroke_pts)
    
    bg_elements = ""
    if bg_curve is not None:
        bx, by = bg_curve
        b_scaled_y = baseline - (by / max_y) * (baseline - top)
        b_scaled_x = left + (bx - bx[0]) / (bx[-1] - bx[0]) * (right - left)
        b_stroke_pts = [f"{sx:.1f},{sy:.1f}" for sx, sy in zip(b_scaled_x, b_scaled_y)]
        b_stroke_d = "M " + " L ".join(b_stroke_pts)
        bg_elements = f'<path d="{b_stroke_d}" fill="none" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="3,2.5" opacity="0.8"/>'
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 48" width="90" height="44" style="display:inline-block; vertical-align:middle; background:#ffffff; border-radius:6px; border:1px solid #e2e8f0; padding:2px;">
  <line x1="4" y1="{baseline}" x2="96" y2="{baseline}" stroke="#cbd5e1" stroke-width="1.5" stroke-linecap="round"/>
  {bg_elements}
  <path d="{fill_d}" fill="{fill_color}"/>
  <path d="{stroke_d}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''
    return svg

def make_discrete_svg(k_vals, probs, color="#2563eb", bar_width=4.5, baseline=42, top=8, left=14, right=86):
    max_p = np.max(probs) if np.max(probs) > 0 else 1.0
    n = len(k_vals)
    if n > 1:
        x_coords = left + (np.array(k_vals) - k_vals[0]) / (k_vals[-1] - k_vals[0]) * (right - left)
    else:
        x_coords = [50]
        
    rects = []
    for x, p in zip(x_coords, probs):
        h = (p / max_p) * (baseline - top)
        y = baseline - h
        rects.append(f'<rect x="{x - bar_width/2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" rx="1.5" fill="{color}"/>')
        rects.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{color}"/>')
        
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 48" width="90" height="44" style="display:inline-block; vertical-align:middle; background:#ffffff; border-radius:6px; border:1px solid #e2e8f0; padding:2px;">
  <line x1="4" y1="{baseline}" x2="96" y2="{baseline}" stroke="#cbd5e1" stroke-width="1.5" stroke-linecap="round"/>
  {"".join(rects)}
</svg>'''
    return svg

# 1. Bernoulli
svg_bern = make_discrete_svg([0, 1], [0.30, 0.70], color="#2563eb", bar_width=9, left=30, right=70)
with open(f"{OUT_DIR}/bernoulli.svg", "w") as f:
    f.write(svg_bern)

# 2. Binomial
k_binom = np.arange(0, 11)
p_binom = stats.binom.pmf(k_binom, 10, 0.45)
svg_binom = make_discrete_svg(k_binom, p_binom, color="#2563eb", bar_width=4.2, left=10, right=90)
with open(f"{OUT_DIR}/binomial.svg", "w") as f:
    f.write(svg_binom)

# 3. Poisson
k_pois = np.arange(0, 10)
p_pois = stats.poisson.pmf(k_pois, 3.5)
svg_pois = make_discrete_svg(k_pois, p_pois, color="#059669", bar_width=4.5, left=12, right=88)
with open(f"{OUT_DIR}/poisson.svg", "w") as f:
    f.write(svg_pois)

# 4. Negative Binomial
k_nb = np.arange(0, 12)
p_nb = stats.nbinom.pmf(k_nb, 1.3, 1.3 / (3.0 + 1.3))
svg_nb = make_discrete_svg(k_nb, p_nb, color="#059669", bar_width=4.0, left=10, right=90)
with open(f"{OUT_DIR}/negative_binomial.svg", "w") as f:
    f.write(svg_nb)

# 5. Hurdle / Zero-Inflated
k_zi = np.arange(0, 9)
p_pos = stats.poisson.pmf(np.arange(1, 9), 3.0)
p_pos = p_pos / np.sum(p_pos) * 0.45
p_zi = np.zeros(9)
p_zi[0] = 0.55
p_zi[1:] = p_pos
svg_zi = make_discrete_svg(k_zi, p_zi, color="#059669", bar_width=4.8, left=12, right=88)
with open(f"{OUT_DIR}/zero_inflated.svg", "w") as f:
    f.write(svg_zi)

# 6. Ordered Categorical
k_ord = np.arange(1, 6)
p_ord = np.array([0.08, 0.22, 0.42, 0.20, 0.08])
svg_ord = make_discrete_svg(k_ord, p_ord, color="#7c3aed", bar_width=8.0, left=16, right=84)
with open(f"{OUT_DIR}/ordered_categorical.svg", "w") as f:
    f.write(svg_ord)

# 7. Categorical / Multinomial
k_cat = np.arange(1, 5)
p_cat = np.array([0.38, 0.16, 0.30, 0.16])
svg_cat = make_discrete_svg(k_cat, p_cat, color="#7c3aed", bar_width=10.0, left=20, right=80)
with open(f"{OUT_DIR}/categorical.svg", "w") as f:
    f.write(svg_cat)

# 8. Normal
x_norm = np.linspace(-3, 3, 60)
y_norm = stats.norm.pdf(x_norm)
svg_norm = make_continuous_svg(x_norm, y_norm, color="#2563eb", fill_color="rgba(37,99,235,0.18)")
with open(f"{OUT_DIR}/normal.svg", "w") as f:
    f.write(svg_norm)

# 9. Student-t
x_t = np.linspace(-3.5, 3.5, 60)
y_t = stats.t.pdf(x_t, df=2.5)
y_norm_bg = stats.norm.pdf(x_t)
svg_t = make_continuous_svg(x_t, y_t, color="#2563eb", fill_color="rgba(37,99,235,0.18)", bg_curve=(x_t, y_norm_bg))
with open(f"{OUT_DIR}/student_t.svg", "w") as f:
    f.write(svg_t)

# 10. Lognormal
x_ln = np.linspace(0.01, 4.5, 70)
y_ln = stats.lognorm.pdf(x_ln, s=0.75)
svg_ln = make_continuous_svg(x_ln, y_ln, color="#d97706", fill_color="rgba(217,119,6,0.18)")
with open(f"{OUT_DIR}/lognormal.svg", "w") as f:
    f.write(svg_ln)

# 11. Gamma
x_gam = np.linspace(0.01, 7.0, 70)
y_gam = stats.gamma.pdf(x_gam, a=2.5, scale=1.0)
svg_gam = make_continuous_svg(x_gam, y_gam, color="#d97706", fill_color="rgba(217,119,6,0.18)")
with open(f"{OUT_DIR}/gamma.svg", "w") as f:
    f.write(svg_gam)

# 12. Exponential
x_exp = np.linspace(0, 4.0, 60)
y_exp = stats.expon.pdf(x_exp, scale=1.2)
svg_exp = make_continuous_svg(x_exp, y_exp, color="#dc2626", fill_color="rgba(220,38,38,0.18)")
with open(f"{OUT_DIR}/exponential.svg", "w") as f:
    f.write(svg_exp)

# 13. Shifted / ex-Gaussian
x_exg = np.linspace(0, 6.0, 80)
tau = 1.0
y_exg = np.where(x_exg < tau, 0.0, stats.exponnorm.pdf(x_exg - tau, K=1.8, loc=0.6, scale=0.5))
svg_exg = make_continuous_svg(x_exg, y_exg, color="#dc2626", fill_color="rgba(220,38,38,0.18)")
with open(f"{OUT_DIR}/ex_gaussian.svg", "w") as f:
    f.write(svg_exg)

print("Generated 13 distribution SVGs in", OUT_DIR)

for d in ["_build/html/_static/dist_icons", "_build/html/build/_static/dist_icons", "_build/site/public/_static/dist_icons"]:
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        if f.endswith(".svg"):
            shutil.copy(os.path.join(OUT_DIR, f), os.path.join(d, f))
print("Synced to all build folders.")
