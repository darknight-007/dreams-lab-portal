#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(open(Path.home() / "maskrcnn_organize" / "maskrcnn_inventory.csv")))

by_fp = defaultdict(list)
for r in rows:
    if r["fingerprint"]:
        by_fp[r["fingerprint"]].append(r)

bkup_files = [r for r in rows if "/dreamslab-hdd-bkup/" in r["path"]]
primary_files = [
    r for r in rows
    if "/zhiang/zhiang_deep_learning/mask_rcnn_pytorch" in r["path"]
    and "/mnt/12tb-hdd-B" in r["path"]
]

bkup_with_primary_twin = 0
bkup_unique = 0
bkup_unique_size = 0
bkup_with_twin_size = 0
for r in bkup_files:
    fp = r["fingerprint"]
    if fp:
        twins = [
            x for x in by_fp[fp]
            if "/zhiang/zhiang_deep_learning/" in x["path"]
            and "/mnt/12tb-hdd-B" in x["path"]
        ]
        if twins:
            bkup_with_primary_twin += 1
            bkup_with_twin_size += int(r["size"])
            continue
    bkup_unique += 1
    bkup_unique_size += int(r["size"])

print(f"bkup-hdd with twin in zhiang-B-primary: {bkup_with_primary_twin}  ({bkup_with_twin_size/1e9:.1f} GB)")
print(f"bkup-hdd UNIQUE (no primary twin):      {bkup_unique}  ({bkup_unique_size/1e9:.1f} GB)")
print()

primary_by_exp = defaultdict(list)
for r in primary_files:
    primary_by_exp[r["experiment"]].append(r)

print("PRIMARY TREE (Zhiang) -- epoch sprawl per experiment:")
print(f"{'experiment':<52} {'files':>5} {'epochs':>6} {'size_GB':>8}")
total_primary = 0
keep_2 = 0
keep_3 = 0
for exp in sorted(primary_by_exp.keys()):
    files = primary_by_exp[exp]
    epochs = sorted({int(f["epoch"]) for f in files if f["epoch"]})
    size = sum(int(f["size"]) for f in files)
    total_primary += size
    by_epoch = sorted(files, key=lambda r: int(r["epoch"]) if r["epoch"] else -1, reverse=True)
    keep_2 += sum(int(f["size"]) for f in by_epoch[:2])
    keep_3 += sum(int(f["size"]) for f in by_epoch[:3])
    name = exp.split("/")[-1]
    print(f"  {name:<50} {len(files):>5} {len(epochs):>6} {size/1e9:>8.2f}")

print()
print(f"Primary total:                              {total_primary/1e9:.1f} GB")
print(f"Keep last 2 epochs/experiment:              {keep_2/1e9:.1f} GB  (reclaim {(total_primary-keep_2)/1e9:.1f} GB)")
print(f"Keep last 3 epochs/experiment:              {keep_3/1e9:.1f} GB  (reclaim {(total_primary-keep_3)/1e9:.1f} GB)")

# Same for bkup tree
print()
print("BACKUP TREE -- epoch sprawl per experiment:")
bkup_by_exp = defaultdict(list)
for r in bkup_files:
    bkup_by_exp[r["experiment"]].append(r)
total_bkup = sum(int(f["size"]) for f in bkup_files)
keep2_bkup = 0
for exp, files in bkup_by_exp.items():
    by_epoch = sorted(files, key=lambda r: int(r["epoch"]) if r["epoch"] else -1, reverse=True)
    keep2_bkup += sum(int(f["size"]) for f in by_epoch[:2])
print(f"Backup total:                               {total_bkup/1e9:.1f} GB")
print(f"Backup keep last 2 epochs/experiment:       {keep2_bkup/1e9:.1f} GB  (reclaim {(total_bkup-keep2_bkup)/1e9:.1f} GB)")
print(f"Backup deleted entirely (all covered):      reclaim {bkup_with_twin_size/1e9:.1f} GB")
