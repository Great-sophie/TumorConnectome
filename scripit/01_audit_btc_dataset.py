#!/usr/bin/env python3
"""Read-only audit of local BTC preoperative/postoperative derivatives.

Outputs: cohort_manifest.csv, connectome_qc.csv, tumor_mask_qc.csv,
         audit_summary.json, audit_summary.txt. No original data is modified.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    from scipy.io import loadmat
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install numpy scipy nibabel") from exc

ID_RE = re.compile(r"^sub-(PAT|CON)(\d+)$", re.IGNORECASE)
THRESHOLDS = (0.0, 0.01, 0.1, 0.5)
SC_KEYS = ("SCthrAn",)
FC_KEYS = ("FC_cc_DK68", "FC_cc_DK68_deconv")


def sorted_ids(items):
    return sorted(items, key=lambda s: (s.split('-')[1][:3], int(re.search(r'(\d+)$', s).group(1))))


def ids_in(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.name for p in folder.iterdir() if p.is_dir() and ID_RE.fullmatch(p.name)}


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def keys_as_strings(mat: dict) -> str:
    return ";".join(k for k in mat if not k.startswith("__"))


def inspect_matrix(path: Path, key: str, kind: str) -> dict:
    record = {"file": str(path), "key": key, "kind": kind, "exists": path.is_file(),
              "status": "missing", "shape": "", "finite": "", "min": "", "max": "",
              "symmetry_max_abs": "", "zero_fraction": "", "diagonal_max_abs": "",
              "diagonal_one_max_abs": "", "negative_fraction": "", "offdiag_nonzero_fraction": "",
              "keys_available": "", "warning": ""}
    if not path.is_file():
        return record
    try:
        data = loadmat(str(path), variable_names=[key])
        if key not in data:
            # Load just metadata on fallback to disclose keys present
            from scipy.io import whosmat
            record.update(status="missing_key", keys_available=";".join(k for k, _, _ in whosmat(str(path))))
            return record
        arr = np.asarray(data[key], dtype=np.float64)
        record["shape"] = "x".join(map(str, arr.shape))
        if arr.shape != (68, 68):
            record.update(status="bad_shape", warning="Expected DK68 68x68 matrix")
            return record
        finite = np.isfinite(arr).all()
        record["finite"] = bool(finite)
        if not finite:
            record.update(status="nonfinite", warning="NaN or infinity present")
            return record
        symerr = float(np.max(np.abs(arr - arr.T)))
        diag = np.diag(arr)
        offdiag = arr[~np.eye(68, dtype=bool)]
        record.update(min=float(arr.min()), max=float(arr.max()), symmetry_max_abs=symerr,
                      zero_fraction=float(np.mean(arr == 0)),
                      diagonal_max_abs=float(np.max(np.abs(diag))),
                      diagonal_one_max_abs=float(np.max(np.abs(diag - 1))),
                      negative_fraction=float(np.mean(arr < 0)),
                      offdiag_nonzero_fraction=float(np.mean(offdiag != 0)))
        problems = []
        if symerr > 1e-6:
            problems.append("asymmetric")
        if kind == "SC":
            if float(arr.min()) < -1e-9:
                problems.append("negative_SC")
            if float(np.max(np.abs(diag))) > 1e-6:
                problems.append("nonzero_SC_diagonal")
        else:
            if float(arr.min()) < -1.000001 or float(arr.max()) > 1.000001:
                problems.append("FC_outside_correlation_range")
            if float(np.max(np.abs(diag - 1))) > 1e-4:
                problems.append("FC_diagonal_not_one")
        record["status"] = "pass" if not problems else "warning"
        record["warning"] = ";".join(problems)
    except Exception as exc:
        record.update(status="error", warning=f"{type(exc).__name__}: {str(exc)[:240]}")
    return record


def inspect_mask(path: Path, subject: str, space: str) -> dict:
    record = {"subject": subject, "space": space, "path": str(path), "exists": path.is_file(),
              "status": "missing", "shape": "", "voxel_size_mm": "", "voxel_volume_mm3": "",
              "min": "", "max": "", "nan_count": "", "inf_count": "", "positive_voxels": "",
              "positive_p25": "", "positive_p50": "", "positive_p75": "", "positive_p99": "",
              "affine_determinant": "", "warning": ""}
    for t in THRESHOLDS:
        label = f"gt_{str(t).replace('.', 'p') }"
        record[f"{label}_voxels"] = ""
        record[f"{label}_ml"] = ""
    if not path.is_file():
        return record
    try:
        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj)
        shape = img.shape
        record["shape"] = "x".join(map(str, shape))
        if len(shape) != 3:
            record.update(status="bad_shape", warning="Expected 3D tumor mask")
            return record
        zooms = img.header.get_zooms()[:3]
        record["voxel_size_mm"] = "x".join(f"{float(z):.6g}" for z in zooms)
        voxel_volume = abs(float(np.linalg.det(img.affine[:3, :3])))
        record["voxel_volume_mm3"] = voxel_volume
        record["affine_determinant"] = float(np.linalg.det(img.affine[:3, :3]))
        record["nan_count"] = int(np.isnan(arr).sum())
        record["inf_count"] = int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            record.update(status="nonfinite", warning="No finite voxels")
            return record
        record["min"] = float(finite.min())
        record["max"] = float(finite.max())
        positive = finite[finite > 0]
        record["positive_voxels"] = int(positive.size)
        if positive.size:
            p25, p50, p75, p99 = np.percentile(positive, [25, 50, 75, 99])
            record.update(positive_p25=float(p25), positive_p50=float(p50),
                          positive_p75=float(p75), positive_p99=float(p99))
        for t in THRESHOLDS:
            label = f"gt_{str(t).replace('.', 'p')}"
            n = int(np.count_nonzero(finite > t))
            record[f"{label}_voxels"] = n
            record[f"{label}_ml"] = n * voxel_volume / 1000.0
        problems = []
        if record["nan_count"] or record["inf_count"]:
            problems.append("nonfinite_values")
        if record["min"] < -1e-6 or record["max"] > 1.00001:
            problems.append("values_outside_0_1")
        if positive.size == 0:
            problems.append("empty_mask")
        if positive.size and np.any((positive > 1e-6) & (positive < 1 - 1e-6)):
            problems.append("continuous_mask_values_threshold_sensitive")
        record["status"] = "pass" if not problems else "warning"
        record["warning"] = ";".join(problems)
    except Exception as exc:
        record.update(status="error", warning=f"{type(exc).__name__}: {str(exc)[:240]}")
    return record


def mask_path(root: Path, sid: str, space: str) -> Path:
    base = root / "derivatives" / "tumor_masks" / sid / "anat"
    return base / f"{sid}_space_{space}_label-tumor.nii"


def main():
    ap = argparse.ArgumentParser(description="Read-only audit of BTC datasets and derivatives")
    ap.add_argument("--root", type=Path, default=Path("/media/sophie/Data2T/MRI Data"),
                    help="Parent directory containing both BTC folders")
    ap.add_argument("--preop", default="Child/openneuro_BTC_preop")
    ap.add_argument("--postop", default="Child/openneuro BTC_postop")
    ap.add_argument("--out", type=Path, default=Path("./btc_audit_results"))
    args = ap.parse_args()
    pre = args.root / args.preop
    post = args.root / args.postop
    if not pre.is_dir() or not post.is_dir():
        ap.error(f"Missing BTC roots: preop={pre} ({pre.exists()}), postop={post} ({post.exists()})")
    out = args.out.expanduser().resolve()
    # Guard against writing the output into original BTC trees
    for source in (pre.resolve(), post.resolve()):
        if out == source or source in out.parents:
            ap.error("--out must be outside the original BTC dataset directories")
    out.mkdir(parents=True, exist_ok=True)

    pre_ids = ids_in(pre / "derivatives" / "TVB")
    post_ids = ids_in(post / "derivatives" / "TVB")
    ids = sorted_ids(pre_ids | post_ids)
    cohorts = []
    connectomes = []
    masks = []
    for sid in ids:
        group = "patient" if "PAT" in sid.upper() else "control"
        row = {"subject": sid, "group": group,
               "preop_TVB": sid in pre_ids, "postop_TVB": sid in post_ids,
               "paired_TVB": sid in pre_ids and sid in post_ids}
        for timepoint, root in (("preop", pre), ("postop", post)):
            ses = root / "derivatives" / "TVB" / sid / f"ses-{timepoint}"
            sc_path = ses / "SCthrAn.mat"
            fc_path = ses / "FC.mat"
            row[f"{timepoint}_SC"] = sc_path.is_file()
            row[f"{timepoint}_FC"] = fc_path.is_file()
            # Determine matrix availability from content, not just filename
            checks = [(sc_path, SC_KEYS[0], "SC"),
                      (fc_path, FC_KEYS[0], "FC"),
                      (fc_path, FC_KEYS[1], "FC_deconv")]
            row[f"{timepoint}_SC_FC_DK68_valid"] = False
            outcomes = {}
            for path, key, kind in checks:
                check = inspect_matrix(path, key, kind)
                check.update(subject=sid, group=group, session=timepoint)
                connectomes.append(check)
                outcomes[key] = check["status"] in ("pass", "warning") and check["finite"] is True and check["shape"] == "68x68"
            row[f"{timepoint}_SC_FC_DK68_valid"] = bool(outcomes.get("SCthrAn") and outcomes.get("FC_cc_DK68"))
        for space in ("T1", "MNI"):
            m = inspect_mask(mask_path(pre, sid, space), sid, space)
            m["group"] = group
            masks.append(m)
            row[f"preop_tumor_mask_{space}"] = m["exists"]
        cohorts.append(row)
        print(f"{sid:12} group={group:7} pre={row['preop_TVB']} post={row['postop_TVB']} "
              f"pre_SCFC={row['preop_SC_FC_DK68_valid']} post_SCFC={row['postop_SC_FC_DK68_valid']}")

    cohort_fields = ["subject", "group", "preop_TVB", "postop_TVB", "paired_TVB",
                     "preop_SC", "preop_FC", "preop_SC_FC_DK68_valid",
                     "postop_SC", "postop_FC", "postop_SC_FC_DK68_valid",
                     "preop_tumor_mask_T1", "preop_tumor_mask_MNI"]
    matrix_fields = ["subject", "group", "session", "kind", "key", "file", "exists", "status", "shape",
                     "finite", "min", "max", "symmetry_max_abs", "zero_fraction", "diagonal_max_abs",
                     "diagonal_one_max_abs", "negative_fraction", "offdiag_nonzero_fraction", "keys_available", "warning"]
    mask_fields = ["subject", "group", "space", "path", "exists", "status", "shape", "voxel_size_mm",
                   "voxel_volume_mm3", "min", "max", "nan_count", "inf_count", "positive_voxels",
                   "positive_p25", "positive_p50", "positive_p75", "positive_p99", "affine_determinant",
                   *[f"gt_{str(t).replace('.', 'p')}_{suffix}" for t in THRESHOLDS for suffix in ("voxels", "ml")],
                   "warning"]
    write_csv(out / "cohort_manifest.csv", cohorts, cohort_fields)
    write_csv(out / "connectome_qc.csv", connectomes, matrix_fields)
    write_csv(out / "tumor_mask_qc.csv", masks, mask_fields)
    pairs = [r for r in cohorts if r["paired_TVB"]]
    valid_pairs = [r for r in pairs if r["preop_SC_FC_DK68_valid"] and r["postop_SC_FC_DK68_valid"]]
    summary = {"preop_tvb_subjects": len(pre_ids), "postop_tvb_subjects": len(post_ids),
               "union_subjects": len(ids), "paired_tvb_subjects": len(pairs),
               "paired_valid_SC_FC_DK68": len(valid_pairs),
               "paired_valid_patient": sum(r["group"] == "patient" for r in valid_pairs),
               "paired_valid_control": sum(r["group"] == "control" for r in valid_pairs),
               "matrix_status_counts": dict(Counter(f"{r['session']}:{r['key']}:{r['status']}" for r in connectomes)),
               "mask_status_counts": dict(Counter(f"{r['space']}:{r['status']}" for r in masks)),
               "note": "Directory presence != quality validation. Mask threshold volumes are sensitivity estimates, not verified ground truth."}
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    readable = ["BTC DATASET AUDIT", "=" * 55,
                f"Preop TVB subject directories: {len(pre_ids)}",
                f"Postop TVB subject directories: {len(post_ids)}",
                f"Unique subjects: {len(ids)}", f"Paired TVB subjects: {len(pairs)}",
                f"Paired subjects with readable 68x68 SC+FC: {len(valid_pairs)}",
                f"  Patients: {summary['paired_valid_patient']}",
                f"  Controls: {summary['paired_valid_control']}",
                "", "WARNING: T1/MNI tumor masks may be interpolated/continuous; do not treat threshold volumes as verified tumor volumes.",
                "WARNING: Validate DK68 node ordering and spatial alignment before tumor-aware regional analysis."]
    (out / "audit_summary.txt").write_text("\n".join(readable) + "\n", encoding="utf-8")
    print("\n" + "\n".join(readable))
    print(f"\nWrote five audit files to: {out}")


if __name__ == "__main__":
    main()
