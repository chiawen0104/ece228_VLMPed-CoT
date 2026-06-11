#!/usr/bin/env python
# evaluate_intent_gemini.py
# ------------------------------------------------------------
# Pedestrian-crossing-intention evaluation with Gemini-2.5-Pro
# ------------------------------------------------------------
# pip install google-generativeai tqdm pillow numpy
# export GOOGLE_API_KEY="YOUR_KEY"
# python evaluate_intent_gemini.py
# ------------------------------------------------------------

import os, re, pickle, gc, time, io
from typing import List
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from google import genai
from google.genai import types

# ════════════════════════════════════════════════════════════
# 1.  数据集
# ════════════════════════════════════════════════════════════
def convert_normalized_to_pixel_coordinates(
    normalized_bboxes: List[List[float]], w: int = 1920, h: int = 1080
) -> List[List[int]]:
    out = []
    for x1, y1, x2, y2 in normalized_bboxes:
        out.append([int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)])
    return out


class IntentDataset(Dataset):
    """
    仅用于推理 / 评测：
      • 不再需要 tokenizer / processor
      • 返回 Python 原生字段便于直观使用
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        speed_dir: str,
        ped_dir: str,
        ped_box_dir: str,
        frame_step: int = 1,
        max_images: int = 15,      # 15 + 场景图 = 16 （Gemini 限制）
        max_samples: int | None = None,
    ):
        self.data = []

        folders = os.listdir(image_dir)[: max_samples or None]

        def extract_number(name: str) -> int:
            nums = re.findall(r"\d+", Path(name).stem)
            return int(nums[0]) if nums else -1

        for folder in folders:
            scene_folder = Path(image_dir) / folder
            label_path   = Path(label_dir) / f"{folder}.pkl"
            speed_path   = Path(speed_dir) / f"{folder}.pkl"
            ped_box_path = Path(ped_box_dir) / f"{folder}.pkl"
            ped_folder   = Path(ped_dir) / folder

            if not (label_path.exists() and speed_path.exists()):
                continue

            # label: 0/1
            with open(label_path, "rb") as f:
                label = pickle.load(f)[0]
            if label not in (0, 1):
                continue

            # speed
            with open(speed_path, "rb") as f:
                speed = (np.array(pickle.load(f)).squeeze()*10).tolist()

            # bounding box
            with open(ped_box_path, "rb") as f:
                ped_box_pixel = convert_normalized_to_pixel_coordinates(pickle.load(f))

            # file lists
            scene_imgs = sorted(
                (scene_folder).glob("*.[jp][pn]g"), key=lambda p: extract_number(p.name)
            )
            ped_imgs = sorted(
                (ped_folder).glob("*.[jp][pn]g"), key=lambda p: extract_number(p.name)
            )

            scene_img  = str(scene_imgs[0])
            ped_imgs   = [str(p) for p in ped_imgs[::frame_step][: max_images]]
            speeds     = speed[::frame_step][: max_images]
            boxes      = ped_box_pixel[::frame_step][: max_images]

            self.data.append(
                dict(
                    scene=scene_img,
                    poses=ped_imgs,
                    speeds=speeds,
                    boxes=boxes,
                    answer="yes" if label == 1 else "no",
                )
            )

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


# ════════════════════════════════════════════════════════════
# 2.  Gemini ⤳ 推理包装
# ════════════════════════════════════════════════════════════
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your key")
assert GOOGLE_API_KEY, "请先在环境变量 GOOGLE_API_KEY 中放入你的 key"
client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL  = "models/gemini-2.5-flash-preview-04-17"
# MODEL  = "models/gemini-2.5-pro-preview-05-06"

def load_img_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
def extract_conclusion(text):
    match = re.search(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", text)
    if match:
        pred = match.group(1).strip()
        return pred
    else:
        raise ValueError("CONCLUSION 标签未找到或格式不正确")

PROMPT_TMPL = (
    "You are an intelligent vision model. Your task is to determine whether a pedestrian has the intention to cross the street within the next 1–2 seconds, based on a sequence of images (approximately 0.5 seconds) captured by an in-vehicle camera and a textual description."
    "The first image is a scene image captured by the in-vehicle camera, with the target pedestrian highlighted by a red bounding box. The remaining images are a sequence showing the pedestrian's continuous poses. Below is the sequence of bounding box coordinates for the target pedestrian in each frame, in the format [x1, y1, x2, y2], where (x1, y1) represents the top-left corner and (x2, y2) the bottom-right corner: {boxes}.The following is the sequence of vehicle speeds corresponding to each frame (unit: km/h): {speeds}."    
    "Please infer the pedestrian’s crossing intention in the next 1–2 seconds based on the information above. Output only ‘Yes’ or ‘No’ as the answer, and do not output anything else.")

def gemini_predict(scene_path: str, pose_paths: list[str],
                   speeds: list[float], boxes: list[list[int]],label:str) -> str:
    contents = [
        PROMPT_TMPL.format(boxes=boxes, speeds=speeds,label=label),
        types.Part.from_bytes(data=load_img_bytes(scene_path),
                              mime_type="image/jpeg"),
    ]
    contents += [
        types.Part.from_bytes(data=load_img_bytes(p),
                              mime_type="image/jpeg")
        for p in pose_paths
    ]

    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
    )
    text = (resp.text or "").strip()
    return text
    # print(text)
    # pred=extract_conclusion(text)



# ════════════════════════════════════════════════════════════
# 3.  数据路径
# ════════════════════════════════════════════════════════════
# When using the JAAD dataset, remember to multiply the speed by 10 at line 79.
ROOT = "/PIE-master/data/pie"
# ROOT ='/JAAD-JAAD_2.0/data/jaad-beh/'
# ROOT ='/JAAD-JAAD_2.0/data/jaad-all/'
paths = dict(
    img = f"{ROOT}/scene image pedestrain bounding box/test",
    lab = f"{ROOT}/label/test",
    spd = f"{ROOT}/speed/test",
    ped = f"{ROOT}/pedestrian image/test",
    box = f"{ROOT}/pedestrain box/test",
)

# ════════════════════════════════════════════════════════════
# 4.  评测
# ════════════════════════════════════════════════════════════
ds = IntentDataset(
    image_dir=paths["img"],
    label_dir=paths["lab"],
    speed_dir=paths["spd"],
    ped_dir=paths["ped"],
    ped_box_dir=paths["box"],
    frame_step=2,
    max_images=15,
    max_samples=None,
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import os
import csv

only_correct = False

# =========================
# 假设以下已定义好：
# - ds: 你的 IntentDataset 数据集
# - gemini_predict: Gemini 推理函数，返回完整文本
# =========================

# 评估过程
gt_labels = []
pred_labels = []
correct = 0
log_rows = []

for idx, sample in enumerate(tqdm(ds, desc="Gemini evaluating")):
    folder_name = Path(sample["scene"]).parent.name
    try:
        text = gemini_predict(sample["scene"], sample["poses"],
                                   sample["speeds"], sample["boxes"],sample["answer"])

        pred_text = text
        pred = "yes" if "yes" in pred_text else "no" if "no" in pred_text else "unrecognized"
    except Exception as e:
        pred = "unrecognized"
        pred_text = f"⚠️ 推理异常：{e}"

    gt = sample["answer"]
    pd = pred

    gt_label = 1 if gt == "yes" else 0
    pd_label = 1 if pd == "yes" else 0

    gt_labels.append(gt_label)
    pred_labels.append(pd_label)

    is_ok = (gt_label == pd_label)
    correct += is_ok

    tqdm.write(f'{idx:02d} | 文件夹: {folder_name} | GT: {gt} | Pred: {pd} | {"✔" if is_ok else "✘"}')

    pred_text_one_line = text.replace("\n", " ").replace("\r", " ").strip()
    if (only_correct and is_ok) or not only_correct:
        log_rows.append([
            idx, folder_name, gt, pd, "✔" if is_ok else "✘", pred_text_one_line
        ])

# =========== 评估指标 ===========
acc = accuracy_score(gt_labels, pred_labels)
prec = precision_score(gt_labels, pred_labels, zero_division=0)
rec = recall_score(gt_labels, pred_labels, zero_division=0)
f1 = f1_score(gt_labels, pred_labels, zero_division=0)

try:
    auc = roc_auc_score(gt_labels, pred_labels)
except ValueError:
    auc = float('nan')

print(f"\n✅ Evaluation results:")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")

# =========== 保存指标和记录 ===========
save_dir = "./metrics"
os.makedirs(save_dir, exist_ok=True)

# 保存 metrics
metrics_path = os.path.join(save_dir, "evaluation_results_without_CoT_pie.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"Evaluation Results\n")
    f.write(f"===================\n")
    f.write(f"Accuracy  : {acc:.4f}\n")
    f.write(f"Precision : {prec:.4f}\n")
    f.write(f"Recall    : {rec:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n")
    f.write(f"AUC       : {auc:.4f}\n")

print(f"\n✅ Metrics saved to {metrics_path}")

# 保存 CSV 日志（包含推理输出）
csv_path = os.path.join(save_dir, "sample_logs_PIE_test.csv")
with open(csv_path, mode="w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "Folder", "GT", "Pred", "Correct", "Gemini_Output"])
    writer.writerows(log_rows)

print(f"📄 Sample logs saved to {csv_path}（仅保留预测正确: {only_correct}）")


