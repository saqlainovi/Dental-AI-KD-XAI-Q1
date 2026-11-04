"""
v4 STAGE 1 - Leak-free multi-center segmentation, external baselines, boundary metrics, paired statistics.

Fixes (Reviewer #3, #4, #7, #8 + self-audit of v3):
  * v3 selected the multi-center checkpoint on the *test* cohorts (`val_dai` used dai_test_loader).
    v4 selects strictly on the DentalAI-val + DENTEX-QE-val cohorts.
  * 44 DENTEX-QE training radiographs (+1 validation) are pixel-level duplicates of DENTEX-QED
    val/test radiographs; they are removed from segmentation training/selection.
  * Adds HD95 / ASSD, per-image paired Wilcoxon tests, bootstrap CIs, and three external baselines
    trained with an identical compact protocol (U-Net/ResNet-34, ResNet-50+All-MLP, ConvNeXt-T+All-MLP).
Outputs -> outputs/v4_reviewer_fixes/
"""
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from v4_common import *  # noqa: F401,F403

SINGLE_CKPT = ROOT / "outputs" / "v2_segmentation" / "checkpoints" / "segformer_b2_tooth_seg_best.pt"


def val_dice(model, df, device):
    return float(eval_seg(model, df, device, "val", tta=False, surface=False)["dice"].mean())


def train_seg(model, name, train_df, val_dai, val_qe, epochs, lr_enc, lr_dec, device, ckpt_path, telemetry):
    loader = DataLoader(MultiCenterSegDataset(train_df, is_train=True), batch_size=4, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": lr_enc, "weight_decay": 1e-2},
        {"params": list(model.decoder.parameters()), "lr": lr_dec, "weight_decay": 1e-2},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = CompositeBceDiceLoss(bce_weight=0.5, dice_weight=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    hist, best = [], -1.0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        torch.cuda.reset_peak_memory_stats()
        tot = 0.0
        opt.zero_grad()
        for b, (x, y, _) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss, _, _ = crit(model(x), y)
                loss = loss / 2.0
            scaler.scale(loss).backward()
            if (b + 1) % 2 == 0 or (b + 1) == len(loader):
                scaler.step(opt); scaler.update(); opt.zero_grad()
            tot += loss.item() * 2.0
        sched.step()
        sec, peak = time.time() - t0, peak_vram_mb()
        vd, vq = val_dice(model, val_dai, device), val_dice(model, val_qe, device)
        comb = 0.5 * (vd + vq)
        hist.append({"epoch": ep, "seconds": round(sec, 2), "train_loss": round(tot / len(loader), 5),
                     "val_dentalai_dice": round(vd, 4), "val_dentex_qe_dice": round(vq, 4),
                     "val_combined_dice": round(comb, 4), "peak_vram_mb": round(peak, 1)})
        print(f"  [{name}] ep {ep:2d}/{epochs} {sec:5.1f}s peak {peak:6.0f}MB loss {tot/len(loader):.4f} "
              f"valDAI {vd:.4f} valQE {vq:.4f}", flush=True)
        if comb > best:
            best = comb
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "val_combined_dice": comb}, ckpt_path)
    pd.DataFrame(hist).to_csv(OUT / f"seg_history_{name}.csv", index=False)
    telemetry.append({"stage": f"seg_{name}", "mean_epoch_s": float(np.mean([h["seconds"] for h in hist])),
                      "total_h": float(np.sum([h["seconds"] for h in hist]) / 3600.0),
                      "peak_vram_mb": float(np.max([h["peak_vram_mb"] for h in hist])), "epochs": epochs})
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    return model


def summarize(df, model_name, cohort):
    row = {"model": model_name, "cohort": cohort, "n_images": len(df)}
    for k in ["dice", "iou", "sensitivity", "specificity", "hd95_px", "assd_px"]:
        m, lo, hi = boot_ci(df[k].values)
        row[f"{k}_mean"], row[f"{k}_ci_low"], row[f"{k}_ci_high"] = m, lo, hi
        row[f"{k}_std"] = float(np.nanstd(df[k].values, ddof=1))
    row["hd95_px_median"] = float(np.nanmedian(df["hd95_px"].values))
    row["n_empty_pred"] = int(df["empty_pred"].sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt-epochs", type=int, default=8)
    ap.add_argument("--baseline-epochs", type=int, default=15)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()
    set_seed()
    dev = get_device()
    m = load_manifest()
    dai_tr, dai_va, dai_te = [m[(m.source == "dentalai") & (m.split == s)] for s in ("train", "val", "test")]
    qe = m[m.source == "dentex_qe"]
    qe_tr = qe[(qe.split == "train") & ~qe.qe_dup_of_qed_valtest]
    qe_va = qe[(qe.split == "val") & ~qe.qe_dup_of_qed_valtest]
    qe_te = qe[qe.split == "test"]
    tufts = m[(m.source == "tufts") & (m.has_teeth == 1)]
    print(f"QE train {len(qe_tr)} (removed {int(((qe.split=='train') & qe.qe_dup_of_qed_valtest).sum())} leaky), "
          f"QE val {len(qe_va)}, QE test {len(qe_te)}, DentalAI test {len(dai_te)}, TUFTS {len(tufts)}", flush=True)
    pool = pd.concat([dai_tr.sample(n=len(qe_tr), random_state=SEED), qe_tr], ignore_index=True)
    telemetry = []

    # ---------- proposed: 60-epoch single-center model adapted on the leak-free multi-center pool ----------
    single = build_seg_model("convnext_tiny_mlp", pretrained=False).to(dev)
    single.load_state_dict(torch.load(SINGLE_CKPT, map_location=dev)["model_state_dict"])
    prop = build_seg_model("convnext_tiny_mlp", pretrained=False).to(dev)
    prop.load_state_dict(single.state_dict())
    prop_ckpt = CKPT / "proposed_multicenter_seg.pt"
    if prop_ckpt.exists():
        prop.load_state_dict(torch.load(prop_ckpt, map_location=dev)["model_state_dict"])
        print("Loaded existing proposed multi-center checkpoint", flush=True)
    else:
        prop = train_seg(prop, "proposed_adapt", pool, dai_va, qe_va, args.adapt_epochs, 3e-5, 1.2e-4, dev, prop_ckpt, telemetry)

    models = {"Single-center ConvNeXt-T+All-MLP (DentalAI, 60 ep)": single,
              "Proposed multi-center ConvNeXt-T+All-MLP (60+8 ep)": prop}

    # ---------- external baselines: identical compact protocol, ImageNet-1K initialisation ----------
    if not args.skip_baselines:
        for arch, label in [("convnext_tiny_mlp", "ConvNeXt-T+All-MLP (compact)"),
                            ("resnet50_mlp", "ResNet-50+All-MLP (compact)"),
                            ("unet_resnet34", "U-Net ResNet-34 (compact)")]:
            ck = CKPT / f"baseline_{arch}.pt"
            mdl = build_seg_model(arch, pretrained=True).to(dev)
            if ck.exists():
                mdl.load_state_dict(torch.load(ck, map_location=dev)["model_state_dict"])
            else:
                set_seed()
                mdl = train_seg(mdl, f"baseline_{arch}", pool, dai_va, qe_va, args.baseline_epochs, 1e-4, 4e-4, dev, ck, telemetry)
            models[f"{label}, {args.baseline_epochs} ep, {mdl.total_params/1e6:.1f}M"] = mdl

    # ---------- evaluation (TTA = horizontal-flip average for every model) ----------
    cohorts = [("DentalAI test", dai_te), ("DENTEX-QE test", qe_te), ("TUFTS zero-shot", tufts)]
    per_img, summ = [], []
    for name, mdl in models.items():
        for cname, cdf in cohorts:
            d = eval_seg(mdl, cdf, dev, cname)
            d["model"] = name
            per_img.append(d)
            summ.append(summarize(d, name, cname))
            print(f"  {name[:55]:55s} | {cname:16s} | Dice {summ[-1]['dice_mean']:.4f} HD95 {summ[-1]['hd95_px_mean']:.1f}px", flush=True)
    per_img = pd.concat(per_img, ignore_index=True)
    per_img.to_csv(OUT / "seg_per_image_v4.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "seg_summary_v4.csv", index=False)

    # ---------- paired Wilcoxon: proposed vs every other model, per cohort ----------
    pname = "Proposed multi-center ConvNeXt-T+All-MLP (60+8 ep)"
    stats = []
    for cname, _ in cohorts:
        pa = per_img[(per_img.model == pname) & (per_img.cohort == cname)].set_index("uid")
        for other in models:
            if other == pname:
                continue
            pb = per_img[(per_img.model == other) & (per_img.cohort == cname)].set_index("uid").loc[pa.index]
            stats.append({"cohort": cname, "proposed": pname, "comparator": other,
                          "dice_diff_mean": float((pa.dice - pb.dice).mean()),
                          "dice_wilcoxon_p": safe_wilcoxon(pa.dice.values, pb.dice.values),
                          "hd95_diff_mean_px": float(np.nanmean(pa.hd95_px.values - pb.hd95_px.values)),
                          "hd95_wilcoxon_p": safe_wilcoxon(pa.hd95_px.values, pb.hd95_px.values)})
    pd.DataFrame(stats).to_csv(OUT / "seg_paired_stats_v4.csv", index=False)

    # ---------- qualitative samples: median and 10th-percentile Dice image per cohort (no cherry-picking) ----------
    samples = {}
    for cname, cdf in cohorts:
        pa = per_img[(per_img.model == pname) & (per_img.cohort == cname)].sort_values("dice").reset_index(drop=True)
        for tag, q in [("median", 0.5), ("p10", 0.10)]:
            uid = pa.iloc[int(q * (len(pa) - 1))]["uid"]
            row = cdf[cdf.uid == uid].iloc[0]
            ds = MultiCenterSegDataset(cdf[cdf.uid == uid], is_train=False)
            x, y, _ = ds[0]
            with torch.no_grad():
                ps = predict_mask_probs(single, x[None].to(dev))[0, 0].cpu().numpy() > 0.5
                pp = predict_mask_probs(prop, x[None].to(dev))[0, 0].cpu().numpy() > 0.5
            img = (x.numpy().transpose(1, 2, 0) * IMG_STD + IMG_MEAN).clip(0, 1)
            samples[f"{cname}|{tag}"] = dict(uid=uid, img=(img * 255).astype(np.uint8), gt=y[0].numpy() > 0.5, single=ps, prop=pp,
                                             dice_single=float(per_img[(per_img.model == list(models)[0]) & (per_img.uid == uid)].dice.iloc[0]),
                                             dice_prop=float(pa[pa.uid == uid].dice.iloc[0]))
    np.save(OUT / "seg_qualitative_samples.npy", samples, allow_pickle=True)
    pd.DataFrame(telemetry).to_csv(OUT / "telemetry_seg_stage.csv", index=False)
    print("SEG STAGE DONE", flush=True)


if __name__ == "__main__":
    main()
