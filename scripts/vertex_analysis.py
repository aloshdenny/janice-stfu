import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from nilearn import datasets, surface
import matplotlib.pyplot as plt

# ── Load all preds ────────────────────────────────────────────────────────────

STUDY_ROOT = Path("./tribe_study")
CATEGORIES = ["porn", "gore", "cute", "nature", "food", "kissing"]

def load_category_mean(category):
    paths = sorted((STUDY_ROOT / category).glob("*/preds.npy"))
    arrays = [np.load(p).mean(axis=0) for p in paths]
    return np.stack(arrays).mean(axis=0)

print("Loading category means...")
means = {cat: load_category_mean(cat) for cat in CATEGORIES}

# ── Contrasts ─────────────────────────────────────────────────────────────────

neutral         = np.stack([means["cute"], means["nature"], means["food"]]).mean(axis=0)
porn_contrast   = means["porn"]    - neutral
gore_contrast   = means["gore"]    - neutral
kiss_contrast   = means["kissing"] - neutral
porn_specific   = porn_contrast    - gore_contrast
gore_specific   = gore_contrast    - porn_contrast
porn_no_romance = porn_contrast    - kiss_contrast
shared_arousal  = np.minimum(porn_contrast, gore_contrast)

CONTRASTS = {
    "Porn contrast":     porn_contrast,
    "Gore contrast":     gore_contrast,
    "Kissing contrast":  kiss_contrast,
    "Porn specific":     porn_specific,
    "Gore specific":     gore_specific,
    "Porn (no romance)": porn_no_romance,
    "Shared arousal":    shared_arousal,
    "Neutral baseline":  neutral,
}

# ── Save vertex masks ─────────────────────────────────────────────────────────

def make_mask(contrast, z_thresh=1.0):
    mu, sd = contrast.mean(), contrast.std()
    return contrast > (mu + z_thresh * sd)

mask_dir = STUDY_ROOT / "masks"
mask_dir.mkdir(exist_ok=True)
for name, data in CONTRASTS.items():
    mask = make_mask(data)
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".npy"
    np.save(mask_dir / fname, mask)
    print(f"  {name:25s}  {mask.sum():5d} vertices  (LH={mask[:10242].sum()}, RH={mask[10242:].sum()})")

print(f"\nMasks saved to {mask_dir}")

# ── Mesh ──────────────────────────────────────────────────────────────────────

fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
lh_coords, lh_faces = surface.load_surf_mesh(fsaverage["pial_left"])
rh_coords, rh_faces = surface.load_surf_mesh(fsaverage["pial_right"])

offset = lh_coords[:, 0].max() - rh_coords[:, 0].min() + 20
rh_coords_offset = rh_coords.copy()
rh_coords_offset[:, 0] += offset

# ── Sulcal depth ──────────────────────────────────────────────────────────────

lh_sulc = surface.load_surf_data(fsaverage["sulc_left"])
rh_sulc = surface.load_surf_data(fsaverage["sulc_right"])

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

lh_sulc = normalize(lh_sulc)
rh_sulc = normalize(rh_sulc)

# ── Blend sulcal gray + activation into vertexcolor ───────────────────────────

hot = plt.get_cmap("hot")

def blend_activation_onto_sulc(sulc_norm, activation, threshold_pct=85):
    thresh = np.nanpercentile(np.abs(activation), threshold_pct)
    vmax   = np.nanpercentile(np.abs(activation), 99)

    r_base = (120 + sulc_norm * 100).astype(float)
    g_base = (120 + sulc_norm * 100).astype(float)
    b_base = (120 + sulc_norm * 100).astype(float)

    colors = []
    for idx in range(len(sulc_norm)):
        val = activation[idx]
        if np.isnan(val) or abs(val) < thresh:
            colors.append(f"rgb({int(r_base[idx])},{int(g_base[idx])},{int(b_base[idx])})")
        else:
            t = float(np.clip((abs(val) - thresh) / (vmax - thresh + 1e-9), 0, 1))
            rc, gc, bc, _ = hot(t)
            colors.append(f"rgb({int(rc*255)},{int(gc*255)},{int(bc*255)})")
    return colors

def make_vertexcolors(data_1d):
    lh_colors = blend_activation_onto_sulc(lh_sulc, data_1d[:10242])
    rh_colors = blend_activation_onto_sulc(rh_sulc, data_1d[10242:])
    return lh_colors, rh_colors

# ── Lighting ──────────────────────────────────────────────────────────────────

lighting = dict(ambient=0.6, diffuse=0.7, specular=0.05, roughness=0.8, fresnel=0.1)
lightposition = dict(x=100, y=200, z=300)

# ── Figure ────────────────────────────────────────────────────────────────────

first_data = list(CONTRASTS.values())[0]
lh_vc_init, rh_vc_init = make_vertexcolors(first_data)

fig = go.Figure()

fig.add_trace(go.Mesh3d(
    x=lh_coords[:, 0], y=lh_coords[:, 1], z=lh_coords[:, 2],
    i=lh_faces[:, 0], j=lh_faces[:, 1], k=lh_faces[:, 2],
    vertexcolor=lh_vc_init,
    showscale=False,
    lighting=lighting, lightposition=lightposition,
    name="LH", hoverinfo="skip",
))

fig.add_trace(go.Mesh3d(
    x=rh_coords_offset[:, 0], y=rh_coords_offset[:, 1], z=rh_coords_offset[:, 2],
    i=rh_faces[:, 0], j=rh_faces[:, 1], k=rh_faces[:, 2],
    vertexcolor=rh_vc_init,
    showscale=False,
    lighting=lighting, lightposition=lightposition,
    name="RH", hoverinfo="skip",
))

# ── Dropdown ──────────────────────────────────────────────────────────────────

dropdown_buttons = []
for cname, cdata in CONTRASTS.items():
    lh_vc, rh_vc = make_vertexcolors(cdata)
    dropdown_buttons.append(dict(
        label=cname,
        method="restyle",
        args=[
            {"vertexcolor": [lh_vc, rh_vc]},
            [0, 1],
        ],
    ))

# ── Layout ────────────────────────────────────────────────────────────────────

fig.update_layout(
    title=dict(text="TRIBE v2 — Contrast Maps", font=dict(color="white", size=16)),
    updatemenus=[dict(
        type="dropdown",
        buttons=dropdown_buttons,
        x=0.0, y=1.1,
        xanchor="left",
        showactive=True,
        bgcolor="#222",
        bordercolor="#555",
        font=dict(color="white"),
    )],
    annotations=[dict(
        text=(
            "<b>Porn contrast</b>: porn − neutral  |  "
            "<b>Gore contrast</b>: gore − neutral  |  "
            "<b>Porn specific</b>: porn − gore  |  "
            "<b>Gore specific</b>: gore − porn  |  "
            "<b>Porn (no romance)</b>: porn − kissing  |  "
            "<b>Shared arousal</b>: both porn+gore"
        ),
        x=0.0, y=-0.06, xref="paper", yref="paper",
        showarrow=False, font=dict(color="#aaa", size=10), align="left",
    )],
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="#0d0d0d",
        camera=dict(eye=dict(x=0, y=-2.0, z=0.5)),
    ),
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(color="white"),
    margin=dict(l=0, r=0, b=100, t=80),
)

out_html = STUDY_ROOT / "contrast_maps.html"
fig.write_html(str(out_html))
print(f"\nSaved → {out_html}")