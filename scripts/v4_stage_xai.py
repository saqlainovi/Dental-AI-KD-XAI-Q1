"""
v4 STAGE 4 - Quantitative Grad-CAM audit with proper baselines (Reviewer #5).

v3 used a fixed central box covering 51.8% of the crop and no baselines, and Fig. 11 labelled crops with
raw `train_*.png` file names.  v4:
  * uses the EXACT ground-truth lesion box inside every crop (from the ROI manifest),
  * reports the expected chance hit-rate (box-area fraction) and a centre-bias baseline,
  * reports an energy-concentration ratio (energy-in-box / box-area fraction; 1.0 == uninformative),
  * runs the Adebayo-et-al. model-randomisation sanity check (re-initialised conv_head + classifier),
  * resamples whole radiographs for 95% CIs,
  * draws the qualitative figure with a fixed-seed random tooth per class (no confidence-based cherry-picking)
    and anonymised labels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from v4_common import *  # noqa: F401,F403

ROI_CKPT = CKPT / "tooth_roi_focal_v4.pt"
FIG_DIRS = [OUT / "figures", ROOT / "Dental_AI_Overleaf_Package" / "figures"]
for d in FIG_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def gradcam(model, x, cls, layer):
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("f", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    model.zero_grad()
    logits = model(x)
    logits[0, cls].backward()
    h1.remove(); h2.remove()
    f, g = store["f"][0].detach().cpu().numpy(), store["g"][0].detach().cpu().numpy()
    cam = np.maximum((g.mean((1, 2))[:, None, None] * f).sum(0), 0.0)
    cam = cv2.resize(cam, (256, 256), interpolation=cv2.INTER_LINEAR)
    return cam / cam.max() if cam.max() > 0 else cam, float(torch.sigmoid(logits)[0, cls].item())


def box_in_crop(r):
    cw, ch = r.crop_x2 - r.crop_x1, r.crop_y2 - r.crop_y1
    x1, y1 = (r.bbox_x - r.crop_x1) / cw * 256, (r.bbox_y - r.crop_y1) / ch * 256
    x2, y2 = (r.bbox_x + r.bbox_w - r.crop_x1) / cw * 256, (r.bbox_y + r.bbox_h - r.crop_y1) / ch * 256
    return [int(np.clip(round(v), 0, 256)) for v in (x1, y1, x2, y2)]


def metrics(cam, box):
    x1, y1, x2, y2 = box
    py, px = np.unravel_index(np.argmax(cam), cam.shape)
    hit = int(x1 <= px < x2 and y1 <= py < y2)
    area = max((x2 - x1) * (y2 - y1), 1) / 65536.0
    energy = float(cam[y1:y2, x1:x2].sum() / max(cam.sum(), 1e-9))
    return hit, energy, area, energy / area


def main():
    set_seed()
    dev = get_device()
    rois = pd.read_csv(ROI_MANIFEST_PATH)
    test = rois[rois.split == "test"].reset_index(drop=True)
    trained = DentalMultiDiseaseClassifier("efficientnet_b2", 4, pretrained=False).to(dev)
    trained.load_state_dict(torch.load(ROI_CKPT, map_location=dev)["model_state_dict"]); trained.eval()
    rand = DentalMultiDiseaseClassifier("efficientnet_b2", 4, pretrained=False).to(dev)
    rand.load_state_dict(trained.state_dict())
    torch.manual_seed(SEED)
    for mod in [rand.backbone.conv_head, rand.backbone.bn2] + list(rand.classifier):
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
    rand.eval()

    yy, xx = np.mgrid[0:256, 0:256]
    center_cam = np.exp(-(((xx - 128) ** 2 + (yy - 128) ** 2) / (2 * (256 / 6) ** 2)))
    rows, vis = [], {}
    for k, r in test.iterrows():
        cls = int(r.primary_class_id)
        im = cv2.cvtColor(cv2.imread(r.roi_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(((im - IMG_MEAN) / IMG_STD).transpose(2, 0, 1)).float()[None].to(dev)
        box = box_in_crop(r)
        cam_t, prob = gradcam(trained, x.clone().requires_grad_(True), cls, trained.backbone.conv_head)
        cam_r, _ = gradcam(rand, x.clone().requires_grad_(True), cls, rand.backbone.conv_head)
        ht, et, area, rt = metrics(cam_t, box)
        hr, er, _, rr = metrics(cam_r, box)
        hc, ec, _, rc = metrics(center_cam, box)
        rows.append({"roi_uid": r.roi_uid, "parent_uid": r.parent_uid, "disease": DISEASES[cls], "prob": prob, "box_area_frac": area,
                     "hit_trained": ht, "energy_trained": et, "ratio_trained": rt,
                     "hit_random_weights": hr, "energy_random_weights": er, "ratio_random_weights": rr,
                     "hit_center_bias": hc, "energy_center_bias": ec, "ratio_center_bias": rc})
        vis[r.roi_uid] = (im, box, cam_t, cam_r, ht, rt, rr)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "xai_per_tooth_v4.csv", index=False)

    out = []
    groups = {u: g.index.values for u, g in d.groupby("parent_uid")}
    keys = list(groups)
    rng = np.random.RandomState(SEED)
    draws = [np.concatenate([groups[keys[j]] for j in rng.randint(0, len(keys), len(keys))]) for _ in range(1000)]
    for name, sub in [(dn, d[d.disease == dn]) for dn in DISEASES] + [("ALL", d)]:
        row = {"disease": name, "n": len(sub), "expected_chance_hit": sub.box_area_frac.mean()}
        for col in ["hit_trained", "hit_random_weights", "hit_center_bias", "energy_trained", "ratio_trained", "ratio_random_weights", "ratio_center_bias"]:
            row[col] = sub[col].mean()
            if name == "ALL":
                vals = [d[col].values[ix].mean() for ix in draws]
                row[col + "_ci_low"], row[col + "_ci_high"] = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
        out.append(row)
    pd.DataFrame(out).to_csv(OUT / "xai_audit_v4.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False), flush=True)

    # ---------- qualitative figure: fixed-seed random tooth per class, anonymised ----------
    rng = np.random.RandomState(7)
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.5))
    for ri, dn in enumerate(DISEASES):
        uid = rng.choice(d[d.disease == dn].roi_uid.values)
        im, box, ct, cr, ht, rt, rr = vis[uid]
        x1, y1, x2, y2 = box
        for ci, (title, cam) in enumerate([(f"{dn}\nCLAHE tooth ROI + exact GT lesion box", None),
                                           (f"Grad-CAM (trained)\nhit={'yes' if ht else 'no'}, conc. ratio={rt:.2f}", ct),
                                           (f"Grad-CAM (randomised weights)\nconc. ratio={rr:.2f}", cr)]):
            ax = axes[ri, ci]
            ax.imshow(im)
            if cam is not None:
                ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#00ffff", lw=2, ls="--"))
            ax.set_title(title, fontsize=9.5, fontweight="bold"); ax.axis("off")
    fig.suptitle("Grad-CAM audit: one randomly drawn test tooth per class (fixed seed, no confidence-based selection)",
                 fontsize=11.5, fontweight="bold", y=0.995)
    plt.tight_layout()
    for fd in FIG_DIRS:
        fig.savefig(fd / "Fig11_gradcam_visualisations.png", dpi=200, bbox_inches="tight")
        fig.savefig(fd / "Fig11_gradcam_visualisations.jpg", dpi=200, bbox_inches="tight")
    print("XAI DONE", flush=True)


if __name__ == "__main__":
    main()
