import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_recall_fscore_support,
                             confusion_matrix, accuracy_score)
from peft import PeftModel, LoraConfig, TaskType
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


model_id = "/data/chl343/VLMPed-CoT-for-Pedestrian-Crossing-Intention-Prediction/LLM-model/Qwen/Qwen2.5-VL-3B-Instruct"

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, model_id="./output/checkpoint-1673_stage2", config=config)
processor = AutoProcessor.from_pretrained(model_id)
model = model.eval()


# ── 工具函數 ──────────────────────────────────────────────────────────────────

def extract_number(filename):
    return int(re.findall(r'\d+', os.path.basename(filename))[0])


def convert_normalized_to_pixel_coordinates(normalized_bboxes,
                                             image_width=1920, image_height=1080):
    pixel_bboxes = []
    for bbox in normalized_bboxes:
        x1, y1, x2, y2 = bbox
        pixel_bboxes.append([
            int(x1 * image_width), int(y1 * image_height),
            int(x2 * image_width), int(y2 * image_height)
        ])
    return pixel_bboxes


def run_batch(batch, gt_labels, predicted_labels, true_labels):
    """送一批資料進模型，收回預測結果"""
    texts  = [b[0] for b in batch]
    images = [b[1] for b in batch]

    processor.tokenizer.padding_side = 'left'
    inputs = processor(
        text=texts, images=images, padding=True, return_tensors="pt"
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=1)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

    for i, output in enumerate(outputs):
        output = output.strip()
        pred = 1 if "yes" in output.lower() else 0 if "no" in output.lower() else None
        if pred is None:
            print(f"[⚠️] 無法識別輸出: {output}")
            continue
        predicted_labels.append(pred)
        true_labels.append(gt_labels[i])


def create_image_turn_duplication(folder_path, speed, pedestrain_dir,
                                   ped_box_pixel, question, dup_idx,
                                   frame_step=2, max_images=16):
    """
    Frame Duplication
    dup_idx: 要複製的幀的 index（0-based，對應 t1~t8）
             None 表示 full sequence
    dup_idx=0 → [t1,t1,t1,t1,t1,t1,t1,t1]
    dup_idx=7 → [t8,t8,t8,t8,t8,t8,t8,t8]
    """
    all_scene_files = sorted([
        os.path.join(folder_path, img)
        for img in os.listdir(folder_path)
        if img.endswith(('.png', '.jpg'))
    ], key=extract_number)

    all_ped_files = sorted([
        os.path.join(pedestrain_dir, img)
        for img in os.listdir(pedestrain_dir)
        if img.endswith(('.png', '.jpg'))
    ], key=extract_number)

    # frame_step=2 取樣 → 8 幀 [t1, t2, t3, t4, t5, t6, t7, t8]
    sampled_ped_files = all_ped_files[::frame_step][:max_images]
    sampled_speed     = speed[::frame_step][:max_images]
    sampled_ped_box   = ped_box_pixel[::frame_step][:max_images]

    n = len(sampled_ped_files)  # 8

    if dup_idx is not None:
        # 將指定幀複製 n 次，取代全部幀
        selected_ped_files = [sampled_ped_files[dup_idx]] * n
        selected_speed     = [sampled_speed[dup_idx]]     * n
        selected_ped_box   = [sampled_ped_box[dup_idx]]   * n
    else:
        selected_ped_files = sampled_ped_files
        selected_speed     = sampled_speed
        selected_ped_box   = sampled_ped_box

    prompt_text = (
        "You are an intelligent vision model. Your task is to determine whether a pedestrian "
        "has the intention to cross the street within the next 1–2 seconds, based on a sequence "
        "of images (approximately 0.5 seconds) captured by an in-vehicle camera and a textual description."
        "The first image is a scene image captured by the in-vehicle camera, with the target pedestrian "
        "highlighted by a red bounding box. The remaining images are a sequence showing the pedestrian's "
        f"continuous poses. Below is the sequence of bounding box coordinates for the target pedestrian "
        f"in each frame, in the format [x1, y1, x2, y2], where (x1, y1) represents the top-left corner "
        f"and (x2, y2) the bottom-right corner: {selected_ped_box}. "
        f"The following is the sequence of vehicle speeds corresponding to each frame (unit: km/h): {selected_speed}."
        " Please reason step-by-step using all of the above information to infer whether the pedestrian "
        "has the intention to cross the street within the next 1–2 seconds. Your reasoning process must "
        "strictly follow the format composed of the following four specific sections: SUMMARY, CAPTION, "
        "REASONING, and CONCLUSION. Use the following format for your reasoning process : "
        "<SUMMARY>[Briefly summarize how you will solve the problem and the steps you will take to reach "
        "the answer.]</SUMMARY> <CAPTION>[Describe the image contents in detail, emphasizing parts relevant "
        "to the question.]</CAPTION> <REASONING>[Step-by-step explanation of your logical reasoning "
        "process.]</REASONING> <CONCLUSION>[Directly give the final answer, must match the correct answer "
        "exactly, directly output the answer 'Yes' or 'No']</CONCLUSION>. Finally, based on the reasoning "
        "results, the model outputs either 'yes' or 'no' as the final answer, which must strictly match "
        "the conclusion derived in the CONCLUSION step, without including any additional content."
    )
    # prompt_text = (
    #     "You are an intelligent vision model. Your task is to determine whether a pedestrian "
    #     "has the intention to cross the street within the next 1–2 seconds, based on a sequence "
    #     "of images (approximately 0.5 seconds) captured by an in-vehicle camera and a textual description."
    #     "The first image is a scene image captured by the in-vehicle camera, with the target pedestrian "
    #     "highlighted by a red bounding box. The remaining images are a sequence showing the pedestrian's "
    #     f"continuous poses. Below is the sequence of bounding box coordinates for the target pedestrian "
    #     f"in each frame, in the format [x1, y1, x2, y2], where (x1, y1) represents the top-left corner "
    #     f"and (x2, y2) the bottom-right corner: {selected_ped_box}. "
    #     f"The following is the sequence of vehicle speeds corresponding to each frame (unit: km/h): {selected_speed}."
    #     " Please infer the pedestrian's crossing intention in the next 1–2 seconds based on the information above."
    #     " Output only 'Yes' or 'No' as the answer, and do not output anything else."
    # )

    messages = [{"type": "image", "image": f"file://{all_scene_files[0]}"}]
    messages.extend([
        {"type": "image", "image": f"file://{img}",
         "resized_height": 300, "resized_width": 150}
        for img in selected_ped_files
    ])
    messages.append({"type": "text", "text": prompt_text + question})
    return messages


# ── 主推論函數 ────────────────────────────────────────────────────────────────

def predict_action_batch(test_image_folder, test_label_folder, test_speed_folder,
                         test_pedestrain_folder, test_boundingbox_folder,
                         question, setting_name, dup_idx,
                         batch_size=4, max_test_samples=None, speed_multiplier=1):
    true_labels = []
    predicted_labels = []

    test_folders = [
        f for f in os.listdir(test_image_folder)
        if os.path.isdir(os.path.join(test_image_folder, f))
        and os.path.exists(os.path.join(test_label_folder, f"{f}.pkl"))
        and os.path.exists(os.path.join(test_speed_folder, f"{f}.pkl"))
    ]
    if max_test_samples:
        test_folders = test_folders[:max_test_samples]

    print(f"\n{'='*60}")
    print(f"Experiment 2B — setting = {setting_name}")
    print(f"Total test samples: {len(test_folders)}")
    print(f"{'='*60}")

    batch           = []
    gt_labels_batch = []

    for folder_name in tqdm(test_folders):
        image_path      = os.path.join(test_image_folder,       folder_name)
        label_path      = os.path.join(test_label_folder,       f"{folder_name}.pkl")
        speed_path      = os.path.join(test_speed_folder,       f"{folder_name}.pkl")
        bounds_path     = os.path.join(test_boundingbox_folder,  f"{folder_name}.pkl")
        pedestrain_path = os.path.join(test_pedestrain_folder,   folder_name)

        with open(label_path, "rb") as f:
            label = pickle.load(f)
        if label not in [[0], [1]]:
            continue
        gt_label = label[0]

        with open(speed_path, "rb") as f:
            speed = (np.array(pickle.load(f)).squeeze() * speed_multiplier).tolist()

        with open(bounds_path, "rb") as f:
            ped_box_pixel = convert_normalized_to_pixel_coordinates(pickle.load(f))

        messages = [{"role": "user", "content": create_image_turn_duplication(
            image_path, speed, pedestrain_path, ped_box_pixel,
            question, dup_idx
        )}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        batch.append((text, image_inputs))
        gt_labels_batch.append(gt_label)

        if len(batch) == batch_size:
            run_batch(batch, gt_labels_batch, predicted_labels, true_labels)
            batch = []
            gt_labels_batch = []

    if batch:
        run_batch(batch, gt_labels_batch, predicted_labels, true_labels)

    # ── 評估 ──
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary"
    )
    auc      = roc_auc_score(true_labels, predicted_labels)
    cm       = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(classification_report(true_labels, predicted_labels, digits=4))
    print(cm)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
          f"F1: {f1_score:.4f} | AUC: {auc:.4f} | Accuracy: {accuracy:.4f}")

    return {
        "setting":          setting_name,
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1_score,
        "auc":              auc,
        "confusion_matrix": cm,
    }


# ── 主執行入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "pie"
    assert dataset in ("pie", "jaad"), f"dataset must be 'pie' or 'jaad', got '{dataset}'"

    data_root = "/data/chl343/VLMPed-CoT-for-Pedestrian-Crossing-Intention-Prediction/pedestrian_data_generation/data"
    base_path = f"{data_root}/pie" if dataset == "pie" else f"{data_root}/jaad-all"

    paths = {
        "image":            f"{base_path}/scene image pedestrain bounding box/test",
        "label":            f"{base_path}/label/test",
        "speed":            f"{base_path}/speed/test",
        "pedestrian_image": f"{base_path}/pedestrian image/test",
        "pedestrian_box":   f"{base_path}/pedestrain box/test",
    }

    speed_multiplier = 10 if dataset == "jaad" else 1

    # setting_name, dup_idx (0-based index 對應 t1~t8)
    settings = [
        ("full",        None),  # t1~t8
        ("duplicate_t1", 0),    # t1 x8
        ("duplicate_t2", 1),    # t2 x8
        ("duplicate_t4", 3),    # t4 x8
        ("duplicate_t8", 7),    # t8 x8
    ]

    all_results = []
    for setting_name, dup_idx in settings:
        metrics = predict_action_batch(
            test_image_folder       = paths["image"],
            test_label_folder       = paths["label"],
            test_speed_folder       = paths["speed"],
            test_pedestrain_folder  = paths["pedestrian_image"],
            test_boundingbox_folder = paths["pedestrian_box"],
            question                = "",
            setting_name            = setting_name,
            dup_idx                 = dup_idx,
            batch_size              = 8,
            # max_test_samples      = 10,
            speed_multiplier        = speed_multiplier,
        )
        all_results.append(metrics)

    # ── Summary Table ──
    full_acc = all_results[0]["accuracy"]
    print("\n" + "="*75)
    print(f"{'Setting':>14} | {'Accuracy':>8} | {'Precision':>9} | "
          f"{'Recall':>6} | {'F1':>6} | {'AUC':>6} | {'Drop':>6}")
    print("-"*75)
    for r in all_results:
        drop = full_acc - r["accuracy"]
        print(f"{r['setting']:>14} | {r['accuracy']:>8.4f} | {r['precision']:>9.4f} | "
              f"{r['recall']:>6.4f} | {r['f1_score']:>6.4f} | {r['auc']:>6.4f} | {drop:>+6.4f}")
    print("="*75)

    # ── 存成 JSON ──
    os.makedirs("./experiments", exist_ok=True)
    output = {
        "experiment": "Experiment 2B: Frame Duplication",
        "full_accuracy": round(full_acc, 4),
        "results": [
            {
                "setting":   r["setting"],
                "accuracy":  round(r["accuracy"],  4),
                "precision": round(r["precision"], 4),
                "recall":    round(r["recall"],    4),
                "f1_score":  round(r["f1_score"],  4),
                "auc":       round(r["auc"],        4),
                "drop":      round(full_acc - r["accuracy"], 4),
                "confusion_matrix": {
                    "tn": int(r["confusion_matrix"][0][0]),
                    "fp": int(r["confusion_matrix"][0][1]),
                    "fn": int(r["confusion_matrix"][1][0]),
                    "tp": int(r["confusion_matrix"][1][1]),
                },
            }
            for r in all_results
        ]
    }
    result_path = f"./experiments/results/exp2b_{dataset}_wo-cot.json"
    with open(result_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Result saved to {result_path}")