# pip install
print("installed")


import os, glob, random, shutil, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import ultralytics
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import Conv, C2f

print("ultralytics:", ultralytics.__version__)
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 0
set_seed(SEED)


ROBOFLOW_API_KEY = "gtKgXl9TDCIuOUhOrWlm"   # get a free key at app.roboflow.com/settings/api
RF_WORKSPACE = "workspace-tedkk"
RF_PROJECT   = "potato-detection-phjcg"
RF_VERSION   = 2

from roboflow import Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
version = project.version(RF_VERSION)
dataset = version.download("yolov8", location="full_data")
print(dataset.location)

DATA_YAML = str((Path("full_data") / "data.yaml").resolve())
with open(DATA_YAML) as f:
    data_cfg = yaml.safe_load(f)

CLASS_NAMES = data_cfg["names"]
NC = data_cfg["nc"]
print(f"Classes ({NC}):", CLASS_NAMES)

for split_key in ["train", "val", "test"]:
    if split_key in data_cfg:
        split_path = Path(data_cfg[split_key])
        n = len(list(split_path.glob("*"))) if split_path.exists() else 0
        print(f"{split_key}: {n} images  ({data_cfg[split_key]})")
    else:
        print(f"{split_key}: not present in this export's data.yaml")

HAS_TEST_SPLIT = "test" in data_cfg
EVAL_SPLIT = "test" if HAS_TEST_SPLIT else "val"
print("\nWill evaluate on split:", EVAL_SPLIT)


class C2f_Attention(C2f):
    """C2f block + global Multi-Head Self-Attention, for the deepest (P5) backbone stage."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, num_heads=4):
        super().__init__(c1, c2, n, shortcut, g, e)
        assert c2 % num_heads == 0, f"C2f_Attention: c2={c2} must be divisible by num_heads={num_heads}"
        self.norm = nn.LayerNorm(c2)
        self.mhsa = nn.MultiheadAttention(embed_dim=c2, num_heads=num_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))  # starts at 0: identical to plain C2f at init

    def forward(self, x):
        y = super().forward(x)                       # (B, C, H, W) standard C2f features
        b, c, h, w = y.shape
        tokens = y.flatten(2).transpose(1, 2)         # (B, H*W, C)
        normed = self.norm(tokens)
        attn_out, _ = self.mhsa(normed, normed, normed)
        tokens = tokens + self.gamma * attn_out       # learnable residual attention
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class GatedFusion(nn.Module):
    """Learnable softmax-gated fusion of two same-resolution feature maps (replaces Concat)."""

    def __init__(self, c1, c2, c_out, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.proj1 = nn.Identity() if c1 == c_out else Conv(c1, c_out, 1, 1)
        self.proj2 = nn.Identity() if c2 == c_out else Conv(c2, c_out, 1, 1)
        self.w = nn.Parameter(torch.ones(2))  # raw gate logits, both start equal

    def forward(self, x):
        x1, x2 = x
        p1, p2 = self.proj1(x1), self.proj2(x2)
        ew = torch.exp(self.w)
        w = ew / (ew.sum() + self.eps)
        return w[0] * p1 + w[1] * p2


tasks.C2f_Attention = C2f_Attention
tasks.GatedFusion = GatedFusion
print("Custom modules registered:", tasks.C2f_Attention, tasks.GatedFusion)


import inspect
import contextlib
import ast
import ultralytics.nn.tasks as tasks

original_source = inspect.getsource(tasks.parse_model)

injection = """
        elif m is C2f_Attention:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
            args.insert(2, n)
            n = 1
        elif m is GatedFusion:
            c1a, c1b = (ch[x] for x in f)
            c2 = args[0]
            args = [c1a, c1b, c2]
"""

patched_source = original_source.replace("        elif m is Concat:", injection.lstrip("\\n") + "        elif m is Concat:")
patched_source = patched_source.replace("def parse_model(", "def parse_model_patched(")

exec(compile(patched_source, "<parse_model_patched>", "exec"), tasks.__dict__)
tasks.parse_model = tasks.__dict__["parse_model_patched"]
print("parse_model dynamically patched for custom layers:", tasks.parse_model)


def make_yaml(nc, use_attention, use_gated, name):
    lines = [
        f"# {name}", f"nc: {nc}", "", "backbone:",
        "  - [-1, 1, Conv,  [16, 3, 2]]                # 0-P1/2",
        "  - [-1, 1, Conv,  [32, 3, 2]]                # 1-P2/4",
        "  - [-1, 1, C2f,   [32, True]]                # 2",
        "  - [-1, 1, Conv,  [64, 3, 2]]                # 3-P3/8",
        "  - [-1, 2, C2f,   [64, True]]                # 4  (P3 skip)",
        "  - [-1, 1, Conv,  [128, 3, 2]]               # 5-P4/16",
        "  - [-1, 2, C2f,   [128, True]]               # 6  (P4 skip)",
        "  - [-1, 1, Conv,  [256, 3, 2]]               # 7-P5/32",
    ]
    if use_attention:
        lines.append("  - [-1, 1, C2f_Attention, [256, True, 1, 0.5, 4]]  # 8  (C2f args + num_heads=4)")
    else:
        lines.append("  - [-1, 1, C2f,   [256, True]]               # 8")
    lines += ["  - [-1, 1, SPPF,  [256, 5]]                  # 9  (P5 skip)", "", "head:",
              "  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]   # 10"]

    def fuse(out_ch, src, idx, tag):
        if use_gated:
            return f"  - [[-1, {src}], 1, GatedFusion, [{out_ch}]]              # {idx}  gated fuse with {tag}"
        return f"  - [[-1, {src}], 1, Concat, [1]]                         # {idx}  concat with {tag}"

    lines.append(fuse(128, 6, 11, "P4"))
    lines.append("  - [-1, 1, C2f, [128, False]]                    # 12  P4'")
    lines += ["", "  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]   # 13"]
    lines.append(fuse(64, 4, 14, "P3"))
    lines.append("  - [-1, 1, C2f, [64, False]]                     # 15  P3' (small objects)")
    lines += ["", "  - [-1, 1, Conv, [64, 3, 2]]                     # 16"]
    lines.append(fuse(128, 12, 17, "P4'"))
    lines.append("  - [-1, 1, C2f, [128, False]]                    # 18  P4'' (medium objects)")
    lines += ["", "  - [-1, 1, Conv, [128, 3, 2]]                    # 19"]
    lines.append(fuse(256, 9, 20, "P5"))
    lines.append("  - [-1, 1, C2f, [256, False]]                    # 21  P5'' (large objects)")
    lines += ["", "  - [[15, 18, 21], 1, Detect, [nc]]               # 22"]
    return "\n".join(lines) + "\n"


VARIANTS = {
    "baseline_yolov8n":     dict(use_attention=False, use_gated=False),
    "attention_only":       dict(use_attention=True,  use_gated=False),
    "gated_fusion_only":    dict(use_attention=False, use_gated=True),
    "gated_attention_full": dict(use_attention=True,  use_gated=True),
}

yaml_paths = {}
for vname, cfg in VARIANTS.items():
    text = make_yaml(NC, cfg["use_attention"], cfg["use_gated"], vname)
    path = f"{vname}.yaml"
    with open(path, "w") as f:
        f.write(text)
    yaml_paths[vname] = path

print("Generated variants:", list(yaml_paths.keys()))
print("\n" + yaml_paths["gated_attention_full"] + " preview:\n")
print(open(yaml_paths["gated_attention_full"]).read())


def count_params_millions(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def measure_gflops(model, imgsz=640, device="cpu"):
    try:
        from thop import profile
        model = model.to(device).eval()
        dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9  # MACs -> FLOPs (approx), reported in GFLOPs
    except Exception as e:
        print("GFLOPs measurement failed:", e)
        return float("nan")

def measure_latency(torch_model, imgsz=640, n_warmup=10, n_runs=50, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = torch_model.to(device).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            m(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            m(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms_per_image = (t1 - t0) / n_runs * 1000
    return ms_per_image, 1000.0 / ms_per_image


EPOCHS = 120
IMGSZ = 640
BATCH = 16
PATIENCE = 20
SEEDS = [0]   # extend to [0, 1, 2] for mean +/- std results (~3x runtime)

all_runs = []       # one row per (variant, seed)
trained_models = {} # keyed by variant name -> last-seed's eval model, for per-class breakdown later

for vname, yaml_path in yaml_paths.items():
    for seed in SEEDS:
        run_name = f"{vname}_seed{seed}"
        print(f"\n{'='*70}\nTraining: {run_name}\n{'='*70}")
        set_seed(seed)

        model = YOLO(yaml_path)
        n_params = count_params_millions(model.model)
        gflops = measure_gflops(model.model, imgsz=IMGSZ, device="cpu")

        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PATIENCE,
            project="runs_ablation",
            name=run_name,
            exist_ok=True,
            seed=seed,
            verbose=True,
            plots=True,
        )

        best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
        eval_model = YOLO(str(best_weights))
        metrics = eval_model.val(data=DATA_YAML, split=EVAL_SPLIT, imgsz=IMGSZ, plots=True)

        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        ms_per_image, fps = measure_latency(eval_model.model, imgsz=IMGSZ)

        all_runs.append({
            "variant": vname, "seed": seed,
            "attention": VARIANTS[vname]["use_attention"],
            "gated_fusion": VARIANTS[vname]["use_gated"],
            "params_M": round(n_params, 2),
            "GFLOPs": round(gflops, 2),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "latency_ms": round(ms_per_image, 2),
            "fps": round(fps, 1),
            "run_dir": str(Path(model.trainer.save_dir)),
            "val_dir": str(metrics.save_dir),
        })
        trained_models[vname] = eval_model

        print(f"{run_name}: mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}  "
              f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  "
              f"{n_params:.2f}M params  {gflops:.2f} GFLOPs  {ms_per_image:.2f} ms/img")


runs_df = pd.DataFrame(all_runs)
runs_df.to_csv("ablation_runs_raw.csv", index=False)

metric_cols = ["params_M", "GFLOPs", "mAP50", "mAP50_95", "precision", "recall", "f1", "latency_ms", "fps"]
agg = runs_df.groupby(["variant", "attention", "gated_fusion"])[metric_cols].agg(["mean", "std"]).reset_index()

def fmt_mean_std(mean, std):
    if pd.isna(std) or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} \u00b1 {std:.4f}"

summary_rows = []
for _, row in agg.iterrows():
    r = {"variant": row["variant"].iloc[0] if hasattr(row["variant"], "iloc") else row["variant"],
         "attention": row["attention"].iloc[0] if hasattr(row["attention"], "iloc") else row["attention"],
         "gated_fusion": row["gated_fusion"].iloc[0] if hasattr(row["gated_fusion"], "iloc") else row["gated_fusion"]}
    for col in metric_cols:
        r[col] = fmt_mean_std(row[(col, "mean")], row[(col, "std")])
    summary_rows.append(r)

ablation_table = pd.DataFrame(summary_rows)
ablation_table.to_csv("ablation_summary.csv", index=False)
print(ablation_table.to_markdown(index=False))
ablation_table


full_model_name = "gated_attention_full"
full_metrics = trained_models[full_model_name].val(data=DATA_YAML, split=EVAL_SPLIT, imgsz=IMGSZ, plots=True)

per_class_rows = []
box = full_metrics.box
ap_class_index = box.ap_class_index
for i, cls_idx in enumerate(ap_class_index):
    name = CLASS_NAMES[int(cls_idx)]
    ap50 = float(box.ap50[i])
    ap5095 = float(box.ap[i])
    p = float(box.p[i]) if hasattr(box, "p") else float("nan")
    r = float(box.r[i]) if hasattr(box, "r") else float("nan")
    f1c = 2 * p * r / (p + r + 1e-9) if hasattr(box, "p") else float("nan")
    per_class_rows.append({"class": name, "AP50": round(ap50, 4), "AP50_95": round(ap5095, 4),
                            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1c, 4)})

per_class_df = pd.DataFrame(per_class_rows)
per_class_df.to_csv("per_class_results.csv", index=False)
print(per_class_df.to_markdown(index=False))
per_class_df

# NOTE: if this cell errors on box.p / box.r, print(vars(box)) or dir(box) to see the
# exact attribute names for your installed ultralytics version -- these have shifted
# across releases (this is the same class of version-mismatch issue as the parse_model fix).


val_dir = Path(full_metrics.save_dir)
print("Validation plots saved to:", val_dir)
for name in ["confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]:
    p = val_dir / name
    if p.exists():
        img = plt.imread(p)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(name)
        plt.show()
    else:
        print(f"(not found: {name})")


plot_df = runs_df.groupby("variant")[["mAP50", "mAP50_95"]].mean().reindex(VARIANTS.keys())

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(plot_df))
ax.bar([i - 0.175 for i in x], plot_df["mAP50"], width=0.35, label="mAP@0.5")
ax.bar([i + 0.175 for i in x], plot_df["mAP50_95"], width=0.35, label="mAP@0.5:0.95")
ax.set_xticks(list(x))
ax.set_xticklabels(plot_df.index, rotation=20, ha="right")
ax.set_ylabel("mAP")
ax.set_title("Ablation study: detection accuracy by architecture variant")
ax.legend()
plt.tight_layout()
plt.savefig("ablation_map_comparison.png", dpi=150)
plt.show()


out_dir = Path("research_results")
out_dir.mkdir(exist_ok=True)

for f in ["ablation_runs_raw.csv", "ablation_summary.csv", "per_class_results.csv", "ablation_map_comparison.png"]:
    if Path(f).exists():
        shutil.copy(f, out_dir / f)

for name in ["confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png"]:
    p = val_dir / name
    if p.exists():
        shutil.copy(p, out_dir / f"full_model_{name}")

shutil.make_archive("research_results", "zip", out_dir)
print("Saved: research_results.zip")

try:
    from google.colab import files
    files.download("research_results.zip")
except ImportError:
    pass

