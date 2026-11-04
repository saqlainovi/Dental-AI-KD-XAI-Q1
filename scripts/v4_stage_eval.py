"""
v4 STAGE 3 - Statistical evaluation (CPU only) of the deployment-faithful dual-scale pipeline.

Reads outputs/v4_reviewer_fixes/cls_inference_arrays.npy (+ ROI-level CSVs) and writes every table/CSV
that the revised manuscript, figures and notebook consume:
  * validation-only calibration: fusion weights (AUC-driven, parsimonious), bootstrap-stabilised thresholds
    for two operating modes (balanced Youden-J and high-sensitivity >=0.90 val recall)
  * test metrics with 1,000-resample 95% CIs, DeLong AUC tests, exact McNemar tests, paired-bootstrap deltas
  * ECE / Brier calibration, reliability bins
  * dual-branch 20-pass MC-Dropout triage (tau* from validation), selective-prediction gap tests,
    error-detection AUC, risk-coverage curves vs entropy and random rejection
  * ROI-level (per-tooth) AUC/AP with radiograph-clustered bootstrap
  * oracle-box leakage diagnostic (what v3 effectively computed)
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
from v4_common import *  # noqa: F401,F403

A = np.load(OUT / "cls_inference_arrays.npy", allow_pickle=True).item()
VAL, TEST = A["val"], A["test"]
W_GRID = np.round(np.linspace(0.0, 0.6, 7), 2)
TH_GRID = np.linspace(0.02, 0.98, 97)
NB = 1000


def fill(r, g):
    return np.where(np.isnan(r), g, r)


def auc_safe(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def calibrate_weights(y, g, r):
    ws = []
    for c in range(4):
        aucs = np.array([auc_safe(y[:, c], (1 - w) * g[:, c] + w * r[:, c]) for w in W_GRID])
        best = np.nanmax(aucs)
        ws.append(float(W_GRID[np.where(aucs >= best - 0.005)[0][0]]))  # smallest weight within 0.005 AUC of the best
    return ws


def _one_threshold(y, p, mode):
    pos, neg = p[y == 1], p[y == 0]
    sens = np.array([(pos >= t).mean() for t in TH_GRID])
    spec = np.array([(neg < t).mean() for t in TH_GRID])
    if mode == "balanced":
        j = sens + spec - 1.0
        return float(TH_GRID[np.where(j >= j.max() - 1e-9)[0]].mean())
    ok = np.where(sens >= 0.90)[0]
    return float(TH_GRID[ok[-1]]) if len(ok) else float(TH_GRID[0])


def stable_thresholds(y, p, mode, n_boot=200, seed=SEED):
    rng = np.random.RandomState(seed)
    out = []
    for c in range(4):
        vals = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(y), len(y))
            if 0 < y[idx, c].sum() < len(idx):
                vals.append(_one_threshold(y[idx, c], p[idx, c], mode))
        out.append(round(float(np.median(vals)), 2))
    return out


def metrics_rows(config, y, p, th, rng_seed=SEED):
    N = len(y)
    rows = []
    rng = np.random.RandomState(rng_seed)
    idxs = [rng.randint(0, N, N) for _ in range(NB)]
    boot = {k: [] for k in ("f1", "auc", "bal")}
    per_c = {c: {"f1": [], "auc": [], "bal": []} for c in range(4)}
    for idx in idxs:
        fs, as_, bs = [], [], []
        for c in range(4):
            yb, pb = y[idx, c], p[idx, c]
            if not (0 < yb.sum() < len(yb)):
                continue
            cn = binary_counts(yb, (pb >= th[c]).astype(int))
            f, a, b = cn["f1"], roc_auc_score(yb, pb), 0.5 * (cn["recall"] + cn["specificity"])
            per_c[c]["f1"].append(f); per_c[c]["auc"].append(a); per_c[c]["bal"].append(b)
            fs.append(f); as_.append(a); bs.append(b)
        if len(fs) == 4:
            boot["f1"].append(np.mean(fs)); boot["auc"].append(np.mean(as_)); boot["bal"].append(np.mean(bs))
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    macro = {k: [] for k in ("precision", "recall", "specificity", "f1", "auc", "ap", "bal")}
    for c, d in enumerate(DISEASES):
        cn = binary_counts(y[:, c], (p[:, c] >= th[c]).astype(int))
        auc, ap = auc_safe(y[:, c], p[:, c]), average_precision_score(y[:, c], p[:, c])
        bal = 0.5 * (cn["recall"] + cn["specificity"])
        rows.append({"config": config, "disease": d, "threshold": th[c], "n_pos": int(y[:, c].sum()), **cn, "auc": auc, "ap": ap, "balanced_acc": bal,
                     "f1_ci_low": ci(per_c[c]["f1"])[0], "f1_ci_high": ci(per_c[c]["f1"])[1],
                     "auc_ci_low": ci(per_c[c]["auc"])[0], "auc_ci_high": ci(per_c[c]["auc"])[1]})
        for k, v in [("precision", cn["precision"]), ("recall", cn["recall"]), ("specificity", cn["specificity"]), ("f1", cn["f1"]),
                     ("auc", auc), ("ap", ap), ("bal", bal)]:
            macro[k].append(v)
    rows.append({"config": config, "disease": "MACRO", "threshold": np.nan, "n_pos": int(y.sum()),
                 "precision": np.mean(macro["precision"]), "recall": np.mean(macro["recall"]), "specificity": np.mean(macro["specificity"]),
                 "f1": np.mean(macro["f1"]), "auc": np.mean(macro["auc"]), "ap": np.mean(macro["ap"]), "balanced_acc": np.mean(macro["bal"]),
                 "f1_ci_low": ci(boot["f1"])[0], "f1_ci_high": ci(boot["f1"])[1],
                 "auc_ci_low": ci(boot["auc"])[0], "auc_ci_high": ci(boot["auc"])[1]})
    return rows


def paired_boot_delta(y, p_a, th_a, p_b, th_b, seed=SEED):
    rng = np.random.RandomState(seed)
    N = len(y)
    d_auc, d_f1 = [], []
    for _ in range(NB):
        idx = rng.randint(0, N, N)
        aa, ab, fa, fb = [], [], [], []
        ok = True
        for c in range(4):
            yb = y[idx, c]
            if not (0 < yb.sum() < N):
                ok = False
                break
            aa.append(roc_auc_score(yb, p_a[idx, c])); ab.append(roc_auc_score(yb, p_b[idx, c]))
            fa.append(binary_counts(yb, (p_a[idx, c] >= th_a[c]).astype(int))["f1"])
            fb.append(binary_counts(yb, (p_b[idx, c] >= th_b[c]).astype(int))["f1"])
        if ok:
            d_auc.append(np.mean(aa) - np.mean(ab)); d_f1.append(np.mean(fa) - np.mean(fb))
    q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(d_auc), q(d_f1)


def main():
    yv, yt = VAL["y"], TEST["y"]
    gv, gt_ = VAL["g_det"], TEST["g_det"]
    rv, rt = fill(VAL["r_det"], gv), fill(TEST["r_det"], gt_)
    ov, ot = VAL["r_oracle"], TEST["r_oracle"]

    w = calibrate_weights(yv, gv, rv)
    w_or = calibrate_weights(yv, gv, ov)
    fuse = lambda g, r, ww: np.stack([(1 - ww[c]) * g[:, c] + ww[c] * r[:, c] for c in range(4)], 1)
    fv, ft = fuse(gv, rv, w), fuse(gt_, rt, w)
    fv_or, ft_or = fuse(gv, ov, w_or), fuse(gt_, ot, w_or)

    th = {
        "global_balanced": stable_thresholds(yv, gv, "balanced"), "global_highsens": stable_thresholds(yv, gv, "highsens"),
        "fused_balanced": stable_thresholds(yv, fv, "balanced"), "fused_highsens": stable_thresholds(yv, fv, "highsens"),
        "oracle_balanced": stable_thresholds(yv, fv_or, "balanced"),
    }
    json.dump({"fusion_weights": dict(zip(DISEASES, w)), "oracle_fusion_weights": dict(zip(DISEASES, w_or)),
               "thresholds": {k: dict(zip(DISEASES, v)) for k, v in th.items()}}, open(OUT / "calibration_v4.json", "w"), indent=2)
    print("fusion weights", dict(zip(DISEASES, w)), flush=True)
    for k, v in th.items():
        print(" ", k, dict(zip(DISEASES, v)), flush=True)

    rows = []
    rows += metrics_rows("A1 Global EfficientNet-B2 (uncalibrated, theta=0.50)", yt, gt_, [0.5] * 4)
    rows += metrics_rows("A2 Global EfficientNet-B2 (balanced, val-calibrated)", yt, gt_, th["global_balanced"])
    rows += metrics_rows("A3 Global EfficientNet-B2 (high-sensitivity, val-calibrated)", yt, gt_, th["global_highsens"])
    rows += metrics_rows("B1 DentalXAI dual-scale, predicted-mask ROIs (balanced)", yt, ft, th["fused_balanced"])
    rows += metrics_rows("B2 DentalXAI dual-scale, predicted-mask ROIs (high-sensitivity)", yt, ft, th["fused_highsens"])
    rows += metrics_rows("D  DIAGNOSTIC oracle GT-box ROIs (label leakage, invalid at deployment)", yt, ft_or, th["oracle_balanced"])
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cls_results_v4.csv", index=False)

    # ---------- significance: fused (B1) vs global (A2), same operating philosophy ----------
    stat_rows = []
    for c, d in enumerate(DISEASES):
        a1, a2, p = delong_paired(yt[:, c], ft[:, c], gt_[:, c])
        pa = (ft[:, c] >= th["fused_balanced"][c]).astype(int); pb = (gt_[:, c] >= th["global_balanced"][c]).astype(int)
        b01, b10, pm = mcnemar_exact(pa == yt[:, c], pb == yt[:, c])
        stat_rows.append({"disease": d, "auc_fused": a1, "auc_global": a2, "delong_p": p, "mcnemar_fused_only_correct": b01,
                          "mcnemar_global_only_correct": b10, "mcnemar_p": pm})
    (dauc, df1) = paired_boot_delta(yt, ft, th["fused_balanced"], gt_, th["global_balanced"])
    stat_rows.append({"disease": "MACRO (paired bootstrap, fused - global)", "auc_fused": dauc[0], "auc_global": dauc[1], "delong_p": dauc[2],
                      "mcnemar_fused_only_correct": df1[0], "mcnemar_global_only_correct": df1[1], "mcnemar_p": df1[2]})
    pd.DataFrame(stat_rows).to_csv(OUT / "cls_stats_v4.csv", index=False)

    # ---------- calibration (deterministic) ----------
    cal, rel = [], []
    for name, P in [("Global EfficientNet-B2", gt_), ("DentalXAI dual-scale (predicted ROIs)", ft)]:
        for c, d in enumerate(DISEASES):
            cal.append({"model": name, "disease": d, "ece10": ece_score(yt[:, c], P[:, c]), "brier": float(np.mean((P[:, c] - yt[:, c]) ** 2))})
            for pm_, ob, n in reliability_bins(yt[:, c], P[:, c], 5):
                rel.append({"model": name, "disease": d, "mean_pred": pm_, "obs_freq": ob, "n": n})
    cdf = pd.DataFrame(cal)
    mac = cdf.groupby("model")[["ece10", "brier"]].mean().reset_index().assign(disease="MACRO")
    pd.concat([cdf, mac], ignore_index=True).to_csv(OUT / "calibration_metrics_v4.csv", index=False)
    pd.DataFrame(rel).to_csv(OUT / "reliability_v4.csv", index=False)

    # ---------- MC-Dropout triage (dual-branch) ----------
    def mc(Ad, ww):
        g, r = Ad["g_mc"], fill(Ad["r_mc"], Ad["g_mc"])
        f = np.stack([(1 - ww[c]) * g[:, :, c] + ww[c] * r[:, :, c] for c in range(4)], -1)  # (N,T,4)
        return f.mean(1), f.var(1).mean(1)

    mp_v, var_v = mc(VAL, w)
    mp_t, var_t = mc(TEST, w)
    th_mc = stable_thresholds(yv, mp_v, "balanced")
    tau = float(np.percentile(var_v, 75))
    pred = (mp_t >= np.array(th_mc)).astype(int)
    lab_acc = (pred == yt).mean(1)
    exact = (pred == yt).all(1).astype(int)
    low = var_t <= tau
    tri = pd.DataFrame({"uid": TEST["uid"], "name": TEST["name"], "mean_var": var_t, "label_acc": lab_acc, "exact": exact,
                        "n_crops": TEST["n_crops"], "stratum": np.where(low, "low", "high")})
    tri.to_csv(OUT / "triage_per_image_v4.csv", index=False)
    def srow(nm, sel):
        return {"stratum": nm, "n": int(sel.sum()), "pct": 100 * sel.mean(), "label_acc_pct": 100 * lab_acc[sel].mean(),
                "exact_match_pct": 100 * exact[sel].mean(), "mean_var": float(var_t[sel].mean())}
    summ = pd.DataFrame([srow("Low uncertainty (routine screening)", low), srow("High uncertainty (priority referral)", ~low),
                         srow("All test radiographs", np.ones(len(low), bool))])
    summ["tau_star"] = tau
    # gap tests
    rng = np.random.RandomState(SEED)
    gaps = []
    for _ in range(NB):
        idx = rng.randint(0, len(low), len(low))
        l, h = idx[low[idx]], idx[~low[idx]]
        if len(l) and len(h):
            gaps.append(lab_acc[l].mean() - lab_acc[h].mean())
    mw = mannwhitneyu(lab_acc[low], lab_acc[~low], alternative="greater").pvalue
    summ["gap_label_acc_pct"] = 100 * (lab_acc[low].mean() - lab_acc[~low].mean())
    summ["gap_ci_low_pct"] = 100 * np.percentile(gaps, 2.5); summ["gap_ci_high_pct"] = 100 * np.percentile(gaps, 97.5)
    summ["mannwhitney_one_sided_p"] = mw
    # error detection AUC (variance vs entropy)
    ent = -(mp_t * np.log(mp_t + 1e-7) + (1 - mp_t) * np.log(1 - mp_t + 1e-7)).mean(1)
    err = 1 - exact
    def boot_auc(score):
        r_ = np.random.RandomState(SEED); v = []
        for _ in range(NB):
            i = r_.randint(0, len(err), len(err))
            if 0 < err[i].sum() < len(i):
                v.append(roc_auc_score(err[i], score[i]))
        return float(roc_auc_score(err, score)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    ed = {"variance": boot_auc(var_t), "entropy": boot_auc(ent)}
    summ["error_detect_auc_variance"] = ed["variance"][0]; summ["error_detect_auc_variance_ci_low"] = ed["variance"][1]
    summ["error_detect_auc_variance_ci_high"] = ed["variance"][2]; summ["error_detect_auc_entropy"] = ed["entropy"][0]
    summ.to_csv(OUT / "triage_summary_v4.csv", index=False)

    # risk-coverage
    rc = []
    N = len(var_t)
    for score_name, s in [("variance", var_t), ("entropy", ent)]:
        order = np.argsort(s, kind="stable")
        for k in range(int(0.2 * N), N + 1):
            rc.append({"score": score_name, "coverage": k / N, "risk": 1 - lab_acc[order[:k]].mean()})
    rc_df = pd.DataFrame(rc)
    rc_df.to_csv(OUT / "risk_coverage_v4.csv", index=False)
    aurc = rc_df.groupby("score")["risk"].mean().to_dict()
    pd.DataFrame([{"aurc_variance": aurc["variance"], "aurc_entropy": aurc["entropy"], "aurc_random": 1 - lab_acc.mean(),
                   "mc_thresholds": json.dumps(dict(zip(DISEASES, th_mc))),
                   "val_mean_var": float(var_v.mean()), "test_mean_var": float(var_t.mean())}]).to_csv(OUT / "triage_aurc_v4.csv", index=False)
    cm = []
    for c, d in enumerate(DISEASES):
        cm.append({"disease": d, "ece10_mc_mean": ece_score(yt[:, c], mp_t[:, c]), "brier_mc_mean": float(np.mean((mp_t[:, c] - yt[:, c]) ** 2))})
    pd.DataFrame(cm).to_csv(OUT / "calibration_mc_mean_v4.csv", index=False)

    # ---------- ROI-level (per-tooth) metrics with radiograph-clustered bootstrap ----------
    def roi_level(csv, tag):
        d = pd.read_csv(OUT / csv)
        out = []
        groups = {u: g.index.values for u, g in d.groupby("parent_uid")}
        keys = list(groups)
        rng_ = np.random.RandomState(SEED)
        draws = [np.concatenate([groups[keys[j]] for j in rng_.randint(0, len(keys), len(keys))]) for _ in range(NB)]
        for c, (dn, col) in enumerate(zip(DISEASES, DISEASE_COLS)):
            y_, p_ = d[col].values, d[f"prob_{dn}"].values
            a = [roc_auc_score(y_[ix], p_[ix]) for ix in draws if 0 < y_[ix].sum() < len(ix)]
            out.append({"set": tag, "disease": dn, "n_teeth": len(d), "n_pos": int(y_.sum()), "auc": auc_safe(y_, p_),
                        "auc_ci_low": float(np.percentile(a, 2.5)), "auc_ci_high": float(np.percentile(a, 97.5)),
                        "ap": float(average_precision_score(y_, p_)), "prevalence": float(y_.mean())})
        mm = pd.DataFrame(out)
        out.append({"set": tag, "disease": "MACRO", "n_teeth": len(d), "n_pos": int(mm.n_pos.sum()), "auc": mm.auc.mean(),
                    "auc_ci_low": np.nan, "auc_ci_high": np.nan, "ap": mm.ap.mean(), "prevalence": np.nan})
        return out
    pd.DataFrame(roi_level("roi_level_test_predicted_crops.csv", "all predicted teeth (diseased vs unannotated)") +
                 roi_level("roi_level_test_gt_boxes.csv", "annotated diseased teeth only (typing)")).to_csv(OUT / "roi_level_v4.csv", index=False)

    cov = pd.read_csv(OUT / "pred_roi_gt_box_coverage.csv")
    cov.groupby("split")[["n_gt_boxes", "n_detected"]].sum().assign(recall=lambda d: d.n_detected / d.n_gt_boxes).to_csv(OUT / "roi_gt_box_recall_v4.csv")
    print(res[res.disease == "MACRO"][["config", "precision", "recall", "specificity", "f1", "auc", "balanced_acc"]].round(4).to_string(index=False))
    print(summ.round(4).to_string(index=False))
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
