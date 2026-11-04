"""
v4 STAGE 2 - Deployment-faithful (no oracle boxes) Tooth-ROI pipeline, dual-branch MC Dropout, probes.

Fixes (Reviewer #1, #6 + self-audit of v3):
  * v3 cropped test ROIs from ground-truth DENTEX boxes.  Those boxes exist ONLY for diseased teeth and
    radiographs without a box received ROI-probability 0, which leaks the label.  v4 derives every ROI
    from the *predicted* tooth mask (distance-transform watershed -> 1.45x context crop from the
    full-resolution CLAHE radiograph) for train / val / test alike.
  * The ROI specialist now also sees healthy teeth (predicted crops that overlap no annotated box:
    pseudo-negatives, <= 8 per training radiograph).  Assumption "unannotated tooth == no pathology" is
    disclosed as a limitation in the manuscript.
  * MC Dropout (T = 20) is executed through BOTH branches (global + ROI) and fused per pass.
  * Peak-VRAM numbers are *measured* in short training probes (no hard-coded values).
Phases: crops | train | infer  (default: all)
"""
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from v4_common import *  # noqa: F401,F403
from run_v3_complete_pipeline import ToothRoiDataset
from prepare_missing_dentex_v3 import apply_clahe

ROI_SIZE = 256
SIDE_FACTOR = 1.45
MIN_COMPONENT_PX = 150
MAX_CROPS = 40
PRED_MANIFEST = OUT / "pred_roi_manifest.csv"
ROI_CKPT = CKPT / "tooth_roi_focal_v4.pt"
GLOBAL_CKPT = ROOT / "outputs" / "v2_disease_classification" / "checkpoints" / "best_disease_classifier.pt"
T_PASSES = 20


# ───────────────────────── predicted-mask → tooth instances → ROI crops ─────────────────────────

def split_instances(mask):
    """Distance-transform watershed on a binary 512x512 tooth mask -> list of (x1,y1,x2,y2,area)."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX:
            keep[lab == i] = 1
    if keep.sum() == 0:
        return []
    dist = cv2.distanceTransform(keep, cv2.DIST_L2, 5)
    peaks = (dist == ndi.maximum_filter(dist, size=13)) & (dist >= 3.0)
    _, pk = cv2.connectedComponents(peaks.astype(np.uint8))
    markers = np.zeros(m.shape, np.int32)
    markers[keep == 0] = 1
    markers[pk > 0] = pk[pk > 0] + 1
    flood = cv2.cvtColor((255 - np.clip(dist / max(dist.max(), 1) * 255, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(flood, markers)
    boxes = []
    for l in np.unique(markers):
        if l <= 1:
            continue
        reg = (markers == l) & (keep == 1)
        a = int(reg.sum())
        if a < MIN_COMPONENT_PX:
            continue
        ys, xs = np.where(reg)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, a])
    # split abnormally wide merged components into vertical strips
    if len(boxes) >= 4:
        wmed = float(np.median([b[2] - b[0] for b in boxes]))
        out = []
        for b in boxes:
            w = b[2] - b[0]
            k = int(round(w / wmed)) if w > 1.8 * wmed else 1
            if k > 1:
                edges = np.linspace(b[0], b[2], k + 1).astype(int)
                for j in range(k):
                    out.append([edges[j], b[1], edges[j + 1], b[3], b[4] // k])
            else:
                out.append(b)
        boxes = out
    boxes.sort(key=lambda b: -b[4])
    return boxes[:MAX_CROPS]


def iou_xyxy(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def phase_crops(dev):
    set_seed()
    m = load_manifest()
    qed = m[m.source == "dentex"].reset_index(drop=True)
    gt = pd.read_csv(ROI_MANIFEST_PATH)
    seg = build_seg_model("convnext_tiny_mlp", pretrained=False).to(dev)
    seg.load_state_dict(torch.load(CKPT / "proposed_multicenter_seg.pt", map_location=dev)["model_state_dict"])
    seg.eval()
    rows, t0 = [], time.time()
    cover = []
    for k, r in qed.iterrows():
        raw = cv2.imread(str(RAW_QED_DIR / r["original_name"]))
        if raw is None:
            continue
        H, W = raw.shape[:2]
        enh = apply_clahe(raw)
        scale = min(512 / W, 512 / H)
        nw, nh = int(W * scale), int(H * scale)
        xo, yo = (512 - nw) // 2, (512 - nh) // 2
        img = cv2.cvtColor(cv2.imread(r["img_path"]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(((img - IMG_MEAN) / IMG_STD).transpose(2, 0, 1)).float()[None].to(dev)
        mask = predict_mask_probs(seg, x)[0, 0].cpu().numpy() > 0.5
        gts = gt[gt.parent_uid == r["uid"]]
        gt_boxes = [(g.bbox_x, g.bbox_y, g.bbox_x + g.bbox_w, g.bbox_y + g.bbox_h) for g in gts.itertuples()]
        detected = np.zeros(len(gt_boxes), dtype=bool)
        for j, (x1, y1, x2, y2, area) in enumerate(split_instances(mask)):
            ox1, oy1, ox2, oy2 = (x1 - xo) / scale, (y1 - yo) / scale, (x2 - xo) / scale, (y2 - yo) / scale
            cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            side = max(ox2 - ox1, oy2 - oy1, 110.0) * SIDE_FACTOR
            cx1, cy1 = max(0, int(round(cx - side / 2))), max(0, int(round(cy - side / 2)))
            cx2, cy2 = min(W, int(round(cx + side / 2))), min(H, int(round(cy + side / 2)))
            if cx2 <= cx1 + 4 or cy2 <= cy1 + 4:
                continue
            crop_uid = f"{r['uid']}_p{j:02d}"
            path = PRED_ROI_DIR / f"{crop_uid}.png"
            cv2.imwrite(str(path), cv2.resize(enh[cy1:cy2, cx1:cx2], (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LINEAR))
            ious = [iou_xyxy((ox1, oy1, ox2, oy2), g) for g in gt_boxes]
            best = max(ious) if ious else 0.0
            vec = np.zeros(4, dtype=int)
            for ii, g in enumerate(gts.itertuples()):
                if ious[ii] >= 0.3:
                    vec |= np.array([g.disease_impacted, g.disease_caries, g.disease_periapical, g.disease_deep_caries], dtype=int)
                    detected[ii] = True
            status = "pos" if best >= 0.3 else ("neg" if best < 0.1 else "amb")
            rows.append({"roi_uid": crop_uid, "parent_uid": r["uid"], "split": r["split"], "roi_path": str(path),
                         "orig_x1": ox1, "orig_y1": oy1, "orig_x2": ox2, "orig_y2": oy2, "area512": area,
                         "best_gt_iou": best, "status": status,
                         **{c: int(v) for c, v in zip(DISEASE_COLS, vec)}})
        cover.append({"parent_uid": r["uid"], "split": r["split"], "n_gt_boxes": len(gt_boxes), "n_detected": int(detected.sum())})
        if (k + 1) % 100 == 0:
            print(f"  crops {k+1}/{len(qed)} -> {len(rows)} predicted ROIs [{time.time()-t0:.0f}s]", flush=True)
    pm = pd.DataFrame(rows)
    pm.to_csv(PRED_MANIFEST, index=False)
    cv_df = pd.DataFrame(cover)
    cv_df.to_csv(OUT / "pred_roi_gt_box_coverage.csv", index=False)
    for sp in ("train", "val", "test"):
        c = cv_df[cv_df.split == sp]
        p = pm[pm.split == sp]
        print(f"  {sp}: {len(p)} predicted ROIs ({len(p)/max(len(c),1):.1f}/image), pos={int((p.status=='pos').sum())}, "
              f"neg={int((p.status=='neg').sum())}, amb={int((p.status=='amb').sum())}; "
              f"GT-box recall {c.n_detected.sum()}/{c.n_gt_boxes.sum()} = {c.n_detected.sum()/max(c.n_gt_boxes.sum(),1):.3f}", flush=True)


# ───────────────────────── ROI specialist training ─────────────────────────

def phase_train(dev, epochs):
    set_seed()
    gt = pd.read_csv(ROI_MANIFEST_PATH)
    pm = pd.read_csv(PRED_MANIFEST)
    gt_tr = gt[gt.split == "train"].copy()
    pm_tr = pm[pm.split == "train"]
    pos_pred = pm_tr[pm_tr.status == "pos"]
    rng = np.random.RandomState(SEED)
    neg_pred = pm_tr[pm_tr.status == "neg"].groupby("parent_uid", group_keys=False).apply(
        lambda d: d.sample(n=min(8, len(d)), random_state=rng.randint(1 << 30)))
    keep_cols = ["roi_uid", "parent_uid", "roi_path"] + DISEASE_COLS
    train_df = pd.concat([gt_tr[keep_cols], pos_pred[keep_cols], neg_pred[keep_cols]], ignore_index=True)
    val_p = pm[(pm.split == "val") & (pm.status != "amb")][keep_cols].reset_index(drop=True)
    print(f"ROI train set: {len(gt_tr)} GT-box crops + {len(pos_pred)} matched predicted crops + {len(neg_pred)} pseudo-negative "
          f"crops = {len(train_df)}; val (predicted, unambiguous) = {len(val_p)}", flush=True)

    w = []
    for _, r in train_df.iterrows():
        if r["disease_periapical"] == 1:
            w.append(4.5)
        elif r["disease_impacted"] == 1 or r["disease_deep_caries"] == 1:
            w.append(1.8)
        else:
            w.append(1.0)
    loader = DataLoader(ToothRoiDataset(train_df, True), batch_size=16, sampler=WeightedRandomSampler(w, len(w), replacement=True),
                        num_workers=2, pin_memory=True)
    vloader = DataLoader(ToothRoiDataset(val_p, False), batch_size=32, shuffle=False, num_workers=2)
    alphas = [min(float((len(train_df) - train_df[c].sum()) / max(train_df[c].sum(), 1)), 4.0) for c in DISEASE_COLS]
    print("focal alpha:", [round(a, 2) for a in alphas], flush=True)
    model = DentalMultiDiseaseClassifier("efficientnet_b2", 4, pretrained=True, dropout_rate=0.3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = MultiLabelFocalLoss(alpha_weights=torch.tensor(alphas, device=dev), gamma=2.0)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    hist, best = [], -1.0
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); torch.cuda.reset_peak_memory_stats(); tot = 0.0
        for x, y, _, _ in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        sched.step(); sec, peak = time.time() - t0, peak_vram_mb()
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y, _, _ in vloader:
                P.append(torch.sigmoid(model(x.to(dev))).cpu().numpy()); Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        aucs = [roc_auc_score(Y[:, c], P[:, c]) if 0 < Y[:, c].sum() < len(Y) else np.nan for c in range(4)]
        macro = float(np.nanmean(aucs))
        hist.append({"epoch": ep, "seconds": round(sec, 2), "train_loss": round(tot / len(loader), 5), "val_roi_macro_auc": round(macro, 4),
                     "val_roi_periapical_auc": round(float(aucs[2]), 4), "peak_vram_mb": round(peak, 1)})
        print(f"  [ROI v4] ep {ep:2d}/{epochs} {sec:5.1f}s peak {peak:6.0f}MB loss {tot/len(loader):.4f} valAUC {macro:.4f} "
              f"(per-class {[round(float(a), 3) for a in aucs]})", flush=True)
        if macro > best:
            best = macro
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "val_roi_macro_auc": macro}, ROI_CKPT)
    pd.DataFrame(hist).to_csv(OUT / "roi_history_v4.csv", index=False)
    pd.DataFrame([{"stage": "roi_focal_v4", "mean_epoch_s": float(np.mean([h["seconds"] for h in hist])),
                   "total_h": float(np.sum([h["seconds"] for h in hist]) / 3600.0),
                   "peak_vram_mb": float(np.max([h["peak_vram_mb"] for h in hist])), "epochs": epochs}]).to_csv(OUT / "telemetry_roi_stage.csv", index=False)


# ───────────────────────── inference with dual-branch MC Dropout ─────────────────────────

def enable_mc_dropout(model):
    model.eval()
    for mod in model.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d)):
            mod.train()


def agg_rois(p):
    """p: (K,4) tooth probabilities -> (4,) radiograph-level ROI evidence (0.65*max + 0.35*mean of top-2)."""
    if len(p) == 0:
        return np.full(4, np.nan, dtype=np.float32)
    if len(p) == 1:
        return p[0]
    s = np.sort(p, axis=0)[::-1]
    return 0.65 * s[0] + 0.35 * s[:2].mean(0)


def load_crops(paths, dev):
    xs = []
    for p in paths:
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        xs.append(((im - IMG_MEAN) / IMG_STD).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(xs)).float().to(dev) if xs else None


@torch.no_grad()
def roi_probs(model, x):
    with torch.autocast("cuda", dtype=torch.float16):
        return torch.sigmoid(model(x).float()).cpu().numpy()


def phase_infer(dev):
    set_seed()
    m = load_manifest()
    qed = m[m.source == "dentex"].reset_index(drop=True)
    pm = pd.read_csv(PRED_MANIFEST)
    gt = pd.read_csv(ROI_MANIFEST_PATH)
    gmodel = DentalMultiDiseaseClassifier("efficientnet_b2", 4, pretrained=False, dropout_rate=0.3).to(dev)
    gmodel.load_state_dict(torch.load(GLOBAL_CKPT, map_location=dev)["model_state_dict"])
    rmodel = DentalMultiDiseaseClassifier("efficientnet_b2", 4, pretrained=False, dropout_rate=0.3).to(dev)
    rmodel.load_state_dict(torch.load(ROI_CKPT, map_location=dev)["model_state_dict"])
    out = {}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for sp in ("val", "test"):
        rows = qed[qed.split == sp].reset_index(drop=True)
        N = len(rows)
        A = dict(uid=rows.uid.values, name=rows.original_name.values,
                 y=rows[DISEASE_COLS].values.astype(int),
                 g_det=np.zeros((N, 4), np.float32), r_det=np.zeros((N, 4), np.float32), r_oracle=np.zeros((N, 4), np.float32),
                 g_mc=np.zeros((N, T_PASSES, 4), np.float32), r_mc=np.zeros((N, T_PASSES, 4), np.float32),
                 n_crops=np.zeros(N, int))
        for i, r in rows.iterrows():
            im = cv2.cvtColor(cv2.imread(r["img_path"]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(((im - IMG_MEAN) / IMG_STD).transpose(2, 0, 1)).float()[None].to(dev)
            gmodel.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                A["g_det"][i] = torch.sigmoid(gmodel(x).float()).cpu().numpy()[0]
            # predicted-mask ROIs (deployment)
            crops = pm[(pm.parent_uid == r["uid"])].sort_values("roi_uid")
            xc = load_crops(crops.roi_path.tolist(), dev)
            A["n_crops"][i] = len(crops)
            rmodel.eval()
            A["r_det"][i] = agg_rois(roi_probs(rmodel, xc)) if xc is not None else np.nan
            # oracle (ground-truth diseased boxes) diagnostic: 0.0 when no box -> reproduces the v3 leakage
            go = gt[gt.parent_uid == r["uid"]]
            xo = load_crops(go.roi_path.tolist(), dev)
            A["r_oracle"][i] = agg_rois(roi_probs(rmodel, xo)) if xo is not None else 0.0
            # dual-branch MC dropout
            enable_mc_dropout(gmodel); enable_mc_dropout(rmodel)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                A["g_mc"][i] = torch.sigmoid(gmodel(x.repeat(T_PASSES, 1, 1, 1)).float()).cpu().numpy()
            for t in range(T_PASSES):
                A["r_mc"][i, t] = agg_rois(roi_probs(rmodel, xc)) if xc is not None else np.nan
        out[sp] = A
        print(f"  inferred {sp}: N={N}, mean crops/img {A['n_crops'].mean():.1f}, images with 0 crops: {int((A['n_crops']==0).sum())} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    peak = peak_vram_mb()
    np.save(OUT / "cls_inference_arrays.npy", out, allow_pickle=True)

    # per-tooth (predicted ROI) deterministic probabilities on the TEST radiographs
    rmodel.eval()
    te = pm[(pm.split == "test") & (pm.status != "amb")].reset_index(drop=True)
    P = np.concatenate([roi_probs(rmodel, load_crops(te.roi_path.iloc[s:s + 64].tolist(), dev)) for s in range(0, len(te), 64)])
    te_out = te.copy()
    for c, d in enumerate(DISEASES):
        te_out[f"prob_{d}"] = P[:, c]
    te_out.to_csv(OUT / "roi_level_test_predicted_crops.csv", index=False)
    # typing of annotated diseased teeth only (GT boxes, test): secondary diagnostic
    gte = gt[gt.split == "test"].reset_index(drop=True)
    Pg = np.concatenate([roi_probs(rmodel, load_crops(gte.roi_path.iloc[s:s + 64].tolist(), dev)) for s in range(0, len(gte), 64)])
    for c, d in enumerate(DISEASES):
        gte[f"prob_{d}"] = Pg[:, c]
    gte.to_csv(OUT / "roi_level_test_gt_boxes.csv", index=False)
    pd.DataFrame([{"stage": "inference_dual_branch_mc_dropout", "peak_vram_mb": peak, "total_s": time.time() - t0, "passes": T_PASSES}]
                 ).to_csv(OUT / "telemetry_infer_stage.csv", index=False)
    print("INFER DONE", flush=True)


def phase_probe(dev):
    """Measured peak VRAM of one training step (10 iterations, random tensors) at the batch sizes used in training."""
    rows = []
    seg = build_seg_model("convnext_tiny_mlp", pretrained=False).to(dev)
    opt = torch.optim.AdamW(seg.parameters(), 1e-4); sc = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        x = torch.randn(4, 3, 512, 512, device=dev); y = (torch.rand(4, 1, 512, 512, device=dev) > 0.5).float()
        with torch.cuda.amp.autocast():
            loss = CompositeBceDiceLoss()(seg(x), y)[0]
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    rows.append({"stage": "probe_seg_convnext_bs4_512", "peak_vram_mb": peak_vram_mb()})
    del seg, opt
    for name, bs, res in [("probe_global_effnetb2_bs8_512", 8, 512), ("probe_roi_effnetb2_bs16_256", 16, 256)]:
        net = DentalMultiDiseaseClassifier("efficientnet_b2", 4, pretrained=False).to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-4); sc = torch.cuda.amp.GradScaler()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            x = torch.randn(bs, 3, res, res, device=dev); y = (torch.rand(bs, 4, device=dev) > 0.5).float()
            with torch.cuda.amp.autocast():
                loss = F.binary_cross_entropy_with_logits(net(x), y)
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
        rows.append({"stage": name, "peak_vram_mb": peak_vram_mb()})
        del net, opt
    pd.DataFrame(rows).to_csv(OUT / "telemetry_probes.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "crops", "train", "infer", "probe"])
    ap.add_argument("--roi-epochs", type=int, default=20)
    a = ap.parse_args()
    dev = get_device()
    if a.phase in ("all", "crops"):
        phase_crops(dev)
    if a.phase in ("all", "train"):
        phase_train(dev, a.roi_epochs)
    if a.phase in ("all", "infer"):
        phase_infer(dev)
    if a.phase in ("all", "probe"):
        phase_probe(dev)
