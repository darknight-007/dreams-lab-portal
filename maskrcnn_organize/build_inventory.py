#!/usr/bin/env python3
"""Build Mask R-CNN model inventory on 192.168.0.232.

Phase 1: scan known roots, collect every candidate weight file with size+mtime.
Phase 2: group by (basename, size) to identify likely duplicates.
Phase 3: hash only one representative per (basename, size) group, and hash all
         files <= 500 MB that are singletons of interest.

Outputs:
  maskrcnn_inventory.csv    - every file
  maskrcnn_experiments.yaml - grouped by experiment directory
  dedupe_candidates.csv     - groups with >=2 members that share (basename, size)
"""
import csv
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOTS = [
    "/mnt/12tb-hdd-A",
    "/mnt/12tb-hdd-B",
    "/mnt/22tb-hdd",
    "/mnt/2tb-ssd-A",
    "/mnt/1tb-ssd-A",
]

WEIGHT_EXTS = {
    ".h5", ".hdf5", ".pth", ".pt", ".ckpt", ".bin", ".onnx",
    ".engine", ".trt", ".pb", ".pkl", ".param", ".safetensors",
    ".weights", ".npz",
}

NAME_RE = re.compile(r"mask[_-]?r[_-]?cnn|maskrcnn|mrcnn", re.I)
EPOCH_RE = re.compile(r"epoch[_-]?(\d+)", re.I)

SKIP_DIR_PARTS = {
    "site-packages", "dist-packages", ".git", "__pycache__",
    "node_modules", "build", "dist",
}


def should_skip(path_parts):
    return any(p in SKIP_DIR_PARTS for p in path_parts)


def is_candidate(path: Path):
    if path.suffix.lower() not in WEIGHT_EXTS:
        return False
    s = str(path).lower()
    # Accept if path contains maskrcnn-ish token OR sits under known mask_rcnn/maskrcnn dir
    return bool(NAME_RE.search(s))


def md5_head_tail(path: Path, size: int):
    """Cheap fingerprint: md5 of first+last 4 MB + exact size.
    Collisions effectively impossible for our purposes.
    """
    h = hashlib.md5()
    chunk = 4 * 1024 * 1024
    try:
        with open(path, "rb") as f:
            h.update(f.read(chunk))
            if size > chunk:
                f.seek(max(0, size - chunk))
                h.update(f.read(chunk))
    except OSError as e:
        return f"ERR:{e}"
    h.update(str(size).encode())
    return h.hexdigest()


def classify_experiment(p: Path):
    """Identify the experiment directory — i.e. the folder grouping a training run."""
    parts = p.parts
    for i, part in enumerate(parts):
        low = part.lower()
        if low.startswith("trained_param") or low.startswith("logs"):
            # for logs/, experiment is the run subdir
            if low == "logs" and i + 1 < len(parts):
                return str(Path(*parts[: i + 2]))
            return str(Path(*parts[: i + 1]))
        if NAME_RE.search(low) and p.parent.name != part:
            pass
    return str(p.parent)


def main():
    rows = []
    scanned = 0
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            parts = Path(dirpath).parts
            if should_skip(parts):
                dirnames[:] = []
                continue
            for name in filenames:
                scanned += 1
                if scanned % 100000 == 0:
                    print(f"  scanned {scanned} files, collected {len(rows)}",
                          file=sys.stderr)
                full = Path(dirpath) / name
                if not is_candidate(full):
                    continue
                try:
                    st = full.lstat()
                except OSError:
                    continue
                if not os.path.isfile(full):
                    continue
                rows.append({
                    "path": str(full),
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                    "basename": name,
                    "ext": full.suffix.lower(),
                    "experiment": classify_experiment(full),
                    "epoch": (EPOCH_RE.search(name).group(1)
                              if EPOCH_RE.search(name) else ""),
                    "fingerprint": "",
                })
    print(f"Total candidates: {len(rows)} (scanned {scanned})", file=sys.stderr)

    # Fingerprint by (basename, size) groups with >=2 members
    groups = defaultdict(list)
    for r in rows:
        groups[(r["basename"], r["size"])].append(r)
    to_fp = [r for grp in groups.values() if len(grp) >= 2 for r in grp]
    # Also fingerprint large singletons for future de-dup
    for r in rows:
        if r["size"] >= 50 * 1024 * 1024 and r not in to_fp:
            to_fp.append(r)
    print(f"Fingerprinting {len(to_fp)} files", file=sys.stderr)

    for i, r in enumerate(to_fp):
        if i % 50 == 0:
            print(f"  fp {i}/{len(to_fp)}", file=sys.stderr)
        r["fingerprint"] = md5_head_tail(Path(r["path"]), r["size"])

    outdir = Path.home() / "maskrcnn_organize"
    outdir.mkdir(exist_ok=True)

    csv_path = outdir / "maskrcnn_inventory.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "path", "size", "mtime", "basename", "ext",
            "experiment", "epoch", "fingerprint",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}", file=sys.stderr)

    # Dedupe report: group by fingerprint (when non-empty) + basename+size as fallback
    by_fp = defaultdict(list)
    for r in rows:
        key = r["fingerprint"] or f"nofp:{r['basename']}:{r['size']}"
        by_fp[key].append(r)
    dups = {k: v for k, v in by_fp.items() if len(v) >= 2}
    dedupe_csv = outdir / "dedupe_candidates.csv"
    with open(dedupe_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_key", "path", "size", "mtime", "experiment"])
        for key, members in sorted(dups.items(),
                                   key=lambda kv: -sum(m["size"] for m in kv[1])):
            for m in members:
                w.writerow([key, m["path"], m["size"], m["mtime"],
                            m["experiment"]])
    print(f"Wrote {dedupe_csv} ({len(dups)} duplicate groups)", file=sys.stderr)

    # Experiments summary
    exp = defaultdict(lambda: {"files": 0, "total_size": 0,
                                "epochs": set(), "frameworks": set()})
    for r in rows:
        e = exp[r["experiment"]]
        e["files"] += 1
        e["total_size"] += r["size"]
        if r["epoch"]:
            e["epochs"].add(int(r["epoch"]))
        if r["ext"] in (".h5", ".hdf5"):
            e["frameworks"].add("keras")
        elif r["ext"] == ".param":
            e["frameworks"].add("mxnet_or_pytorch_param")
        elif r["ext"] in (".pth", ".pt"):
            e["frameworks"].add("pytorch")
        elif r["ext"] == ".ckpt":
            e["frameworks"].add("tf_ckpt")
        else:
            e["frameworks"].add(r["ext"])

    exp_yaml = outdir / "maskrcnn_experiments.yaml"
    with open(exp_yaml, "w") as f:
        f.write("experiments:\n")
        for path, info in sorted(exp.items(),
                                  key=lambda kv: -kv[1]["total_size"]):
            epochs = sorted(info["epochs"])
            f.write(f"  - path: {path!r}\n")
            f.write(f"    files: {info['files']}\n")
            f.write(f"    total_size_mb: {info['total_size'] / 1e6:.1f}\n")
            f.write(f"    frameworks: {sorted(info['frameworks'])}\n")
            if epochs:
                f.write(f"    epoch_count: {len(epochs)}\n")
                f.write(f"    epoch_range: [{epochs[0]}, {epochs[-1]}]\n")
    print(f"Wrote {exp_yaml} ({len(exp)} experiments)", file=sys.stderr)


if __name__ == "__main__":
    main()
