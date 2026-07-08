import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nilearn import datasets, surface
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# ── Auto-discover categories from tribe_study ─────────────────────────────────

STUDY_ROOT = Path("./tribe_study")
MAX_TRS = 30

def discover_categories():
    cats = []
    for d in sorted(STUDY_ROOT.iterdir()):
        if d.is_dir() and d.name != "masks" and any(d.glob("*/preds.npy")):
            cats.append(d.name)
    return cats

CATEGORIES = discover_categories()
print(f"Discovered {len(CATEGORIES)} categories: {CATEGORIES}")

def load_category_mean(category):
    paths = sorted((STUDY_ROOT / category).glob("*/preds.npy"))
    arrays = []
    for p in paths:
        d = np.load(p)
        d = d[:MAX_TRS]
        arrays.append(d.mean(axis=0))
    return np.stack(arrays).mean(axis=0)

print("Loading category means...")
means = {cat: load_category_mean(cat) for cat in CATEGORIES}
MAPS = {cat: means[cat] for cat in CATEGORIES}
print(f"\n{len(MAPS)} per-class maps generated")

# ── Save vertex masks (native fsaverage5 resolution) ───────────────────────────

def make_mask(data, z_thresh=1.0):
    mu, sd = data.mean(), data.std()
    return data > (mu + z_thresh * sd)

mask_dir = STUDY_ROOT / "masks"
mask_dir.mkdir(exist_ok=True)
for name, data in MAPS.items():
    mask = make_mask(data)
    fname = name.lower().replace(" ", "_") + ".npy"
    np.save(mask_dir / fname, mask)
    print(f"  {name:30s}  {mask.sum():5d} vertices  (LH={mask[:10242].sum()}, RH={mask[10242:].sum()})")
print(f"\nMasks saved to {mask_dir}")

# ── Meshes: full-resolution fsaverage for display, fsaverage5 for data ─────────

fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
fsaverage5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

lh_coords, lh_faces = surface.load_surf_mesh(fsaverage["pial_left"])
rh_coords, rh_faces = surface.load_surf_mesh(fsaverage["pial_right"])
N_LH = lh_coords.shape[0]

lh_sulc = surface.load_surf_data(fsaverage["sulc_left"])
rh_sulc = surface.load_surf_data(fsaverage["sulc_right"])

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

lh_sulc = normalize(lh_sulc)
rh_sulc = normalize(rh_sulc)

# Meshes for fsaverage5 (interactive HTML)
lh_coords5, lh_faces5 = surface.load_surf_mesh(fsaverage5["pial_left"])
rh_coords5, rh_faces5 = surface.load_surf_mesh(fsaverage5["pial_right"])
N_LH5 = lh_coords5.shape[0]

lh_sulc5 = normalize(surface.load_surf_data(fsaverage5["sulc_left"]))
rh_sulc5 = normalize(surface.load_surf_data(fsaverage5["sulc_right"]))

# ── Smooth upsampling: inverse-distance-weighted k-NN on the sphere ───────────
# (nearest-neighbor duplication produces hard blocky plateaus; IDW blends the
# k closest low-res vertices for a continuous-looking gradient instead.)

lh5_sphere, _ = surface.load_surf_mesh(fsaverage5["sphere_left"])
rh5_sphere, _ = surface.load_surf_mesh(fsaverage5["sphere_right"])
lh_sphere, _ = surface.load_surf_mesh(fsaverage["sphere_left"])
rh_sphere, _ = surface.load_surf_mesh(fsaverage["sphere_right"])

K = 8

lh_tree = cKDTree(lh5_sphere)
rh_tree = cKDTree(rh5_sphere)
lh_dist, lh_idx = lh_tree.query(lh_sphere, k=K)
rh_dist, rh_idx = rh_tree.query(rh_sphere, k=K)

def idw_weights(dist, power=2, eps=1e-6):
    w = 1.0 / (dist ** power + eps)
    return w / w.sum(axis=1, keepdims=True)

lh_w = idw_weights(lh_dist)
rh_w = idw_weights(rh_dist)

def upsample(data_1d):
    lh5 = data_1d[:10242]
    rh5 = data_1d[10242:]
    lh_full = (lh5[lh_idx] * lh_w).sum(axis=1)
    rh_full = (rh5[rh_idx] * rh_w).sum(axis=1)
    return np.concatenate([lh_full, rh_full])

print("Upsampling activation maps (smooth IDW) to full-resolution mesh...")
MAPS_FULL = {cat: upsample(data) for cat, data in MAPS.items()}

# ── Blend activation onto sulcal gray ────────────────────────────────────────
# NOTE: base RGB range pushed up toward white (was 120-220, now ~200-255)

hot = plt.get_cmap("hot")

def blend_activation_onto_sulc(sulc_norm, activation, threshold_pct=85):
    thresh = np.nanpercentile(np.abs(activation), threshold_pct)
    vmax   = np.nanpercentile(np.abs(activation), 99)
    r_base = (200 + sulc_norm * 55).astype(float)
    g_base = (200 + sulc_norm * 55).astype(float)
    b_base = (200 + sulc_norm * 55).astype(float)
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

def make_vertexcolors(data_1d_full):
    lh_colors = blend_activation_onto_sulc(lh_sulc, data_1d_full[:N_LH])
    rh_colors = blend_activation_onto_sulc(rh_sulc, data_1d_full[N_LH:])
    return lh_colors, rh_colors

def make_vertexcolors5(data_1d_fsa5):
    lh_colors = blend_activation_onto_sulc(lh_sulc5, data_1d_fsa5[:N_LH5])
    rh_colors = blend_activation_onto_sulc(rh_sulc5, data_1d_fsa5[N_LH5:])
    return lh_colors, rh_colors

# ── Lighting ──────────────────────────────────────────────────────────────────
# ambient bumped up slightly to keep the brighter base color from looking flat

lighting = dict(ambient=0.75, diffuse=0.7, specular=0.05, roughness=0.8, fresnel=0.1)
lightposition = dict(x=100, y=200, z=300)

# ── True sagittal cameras ───────────────────────────────────────────────────
# x = left(-)/right(+), y = posterior(-)/anterior(+), z = inferior(-)/superior(+)
# Lateral (outer) surface of LH faces -x; lateral surface of RH faces +x.
# Orthographic projection avoids perspective "squashing".

lh_camera = dict(
    eye=dict(x=-2.4, y=0.0, z=0.05),
    up=dict(x=0, y=0, z=1),
    projection=dict(type="orthographic"),
)
rh_camera = dict(
    eye=dict(x=2.4, y=0.0, z=0.05),
    up=dict(x=0, y=0, z=1),
    projection=dict(type="orthographic"),
)

def scene_kwargs(camera, domain_x):
    return dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="#0d0d0d",
        aspectmode="data",   # <- prevents squashing, true proportions
        camera=camera,
        domain=dict(x=domain_x, y=[0, 1]),  # <- pins each scene to an exact half
    )

# ── Interactive figure (two side-by-side sagittal scenes) ──────────────────────

first_data = list(MAPS.values())[0]
lh_vc_init, rh_vc_init = make_vertexcolors5(first_data)

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=("Left hemisphere (lateral)", "Right hemisphere (lateral)"),
    horizontal_spacing=0.0,  # <- removes the empty middle strip
)

fig.add_trace(go.Mesh3d(
    x=lh_coords5[:, 0], y=lh_coords5[:, 1], z=lh_coords5[:, 2],
    i=lh_faces5[:, 0], j=lh_faces5[:, 1], k=lh_faces5[:, 2],
    vertexcolor=lh_vc_init,
    showscale=False,
    lighting=lighting, lightposition=lightposition,
    name="LH", hoverinfo="skip",
), row=1, col=1)

fig.add_trace(go.Mesh3d(
    x=rh_coords5[:, 0], y=rh_coords5[:, 1], z=rh_coords5[:, 2],
    i=rh_faces5[:, 0], j=rh_faces5[:, 1], k=rh_faces5[:, 2],
    vertexcolor=rh_vc_init,
    showscale=False,
    lighting=lighting, lightposition=lightposition,
    name="RH", hoverinfo="skip",
), row=1, col=2)

# ── Dropdown ──────────────────────────────────────────────────────────────────

dropdown_buttons = []
for cname, cdata in MAPS.items():
    lh_vc, rh_vc = make_vertexcolors5(cdata)
    dropdown_buttons.append(dict(
        label=cname,
        method="restyle",
        args=[{"vertexcolor": [lh_vc, rh_vc]}, [0, 1]],
    ))

# ── Layout ────────────────────────────────────────────────────────────────────

n_cats = len(CATEGORIES)
fig.update_layout(
    title=dict(
        text=f"TRIBE v2 — Per-Class Activation Maps, Sagittal View ({n_cats} categories)",
        font=dict(color="white", size=16),
    ),
    updatemenus=[dict(
        type="dropdown",
        buttons=dropdown_buttons,
        x=0.0, y=1.12,
        xanchor="left",
        showactive=True,
        bgcolor="#222",
        bordercolor="#555",
        font=dict(color="white"),
    )],
    scene=scene_kwargs(lh_camera, [0.0, 0.5]),
    scene2=scene_kwargs(rh_camera, [0.5, 1.0]),
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(color="white"),
    margin=dict(l=0, r=0, b=40, t=90),
)

out_html = STUDY_ROOT / "per_class_maps_sagittal.html"
fig.write_html(str(out_html))
print(f"\nSaved → {out_html}")

# ── Save static snapshots at max quality ───────────────────────────────────────

class_dir = STUDY_ROOT / "class_maps_sagittal"
class_dir.mkdir(exist_ok=True)

for cname, cdata in MAPS_FULL.items():

    lh_vc, rh_vc = make_vertexcolors(cdata)

    snap = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.0,  # <- removes the empty middle strip
    )

    snap.add_trace(go.Mesh3d(
        x=lh_coords[:, 0], y=lh_coords[:, 1], z=lh_coords[:, 2],
        i=lh_faces[:, 0], j=lh_faces[:, 1], k=lh_faces[:, 2],
        vertexcolor=lh_vc,
        lighting=lighting, lightposition=lightposition,
        hoverinfo="skip", showscale=False,
    ), row=1, col=1)

    snap.add_trace(go.Mesh3d(
        x=rh_coords[:, 0], y=rh_coords[:, 1], z=rh_coords[:, 2],
        i=rh_faces[:, 0], j=rh_faces[:, 1], k=rh_faces[:, 2],
        vertexcolor=rh_vc,
        lighting=lighting, lightposition=lightposition,
        hoverinfo="skip", showscale=False,
    ), row=1, col=2)

    snap.update_layout(
        scene=scene_kwargs(lh_camera, [0.0, 0.5]),
        scene2=scene_kwargs(rh_camera, [0.5, 1.0]),
        paper_bgcolor="black",
        margin=dict(l=0, r=0, t=0, b=0),
        width=3200,
        height=1600,
        showlegend=False,
    )

    snap.add_annotation(
        text=f"<b>{cname}</b>",
        x=0.985, y=0.02,
        xref="paper", yref="paper",
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=44, color="white"),
    )

    fname = cname.lower().replace(" ", "_") + ".png"
    out_file = class_dir / fname

    snap.write_image(
        str(out_file),
        scale=4,  # max res
        engine="kaleido"
    )

    print("Saved:", out_file)