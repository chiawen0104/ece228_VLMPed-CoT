import os
import re
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
from peft import PeftModel, LoraConfig, TaskType
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = "/Qwen/Qwen2.5-VL-3B-Instruct"



model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto", trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_id)

model=model.eval()

# 工具函数
def extract_number(filename):
    return int(re.findall(r'\d+', os.path.basename(filename))[0])

def convert_normalized_to_pixel_coordinates(normalized_bboxes, image_width=1920, image_height=1080):
    pixel_bboxes = []
    for bbox in normalized_bboxes:
        x1, y1, x2, y2 = bbox
        pixel_bbox = [
            int(x1 * image_width),
            int(y1 * image_height),
            int(x2 * image_width),
            int(y2 * image_height)
        ]
        pixel_bboxes.append(pixel_bbox)
    return pixel_bboxes

def create_image_turn(folder_path, speed, pedestrain_dir, ped_box_pixel, question, frame_step=1, max_images=16):
    image_files = sorted([
        os.path.join(folder_path, img)
        for img in os.listdir(folder_path)
        if img.endswith(('.png', '.jpg'))
    ], key=extract_number)
    pedestrain_files = sorted([
        os.path.join(pedestrain_dir, img)
        for img in os.listdir(pedestrain_dir)
        if img.endswith(('.png', '.jpg'))
    ], key=extract_number)

    image_files = image_files[::frame_step][:max_images]
    pedestrain_files = pedestrain_files[::frame_step][:max_images]
    sampled_speed = speed[::frame_step][:max_images]
    ped_box_pixel = ped_box_pixel[::frame_step][:max_images]

    speed_text = (
        f"You are an intelligent vision model. Your task is to determine whether a pedestrian has the intention to cross the street within the next 1–2 seconds, based on a sequence of images (approximately 0.5 seconds) captured by an in-vehicle camera and a textual description."
        f"The first image is a scene image captured by the in-vehicle camera, with the target pedestrian highlighted by a red bounding box. The remaining images are a sequence showing the pedestrian's continuous poses. Below is the sequence of bounding box coordinates for the target pedestrian in each frame, in the format [x1, y1, x2, y2], where (x1, y1) represents the top-left corner and (x2, y2) the bottom-right corner: {ped_box_pixel}.The following is the sequence of vehicle speeds corresponding to each frame (unit: km/h): {sampled_speed}."
        f"Please infer the pedestrian’s crossing intention in the next 1–2 seconds based on the information above. Output only ‘Yes’ or ‘No’ as the answer, and do not output anything else.")

    full_prompt = speed_text + question

    messages = [{"type": "image", "image": f"file://{img}"} for img in [image_files[0]]]
    # mid_idx = len(image_files) // 2
    # messages = [
    #     {"type": "image", "image": f"file://{image_files[0]}"},
    #     {"type": "image", "image": f"file://{image_files[mid_idx]}"},
    #     {"type": "image", "image": f"file://{image_files[-1]}"},
    # ]
    # messages.extend([{"type": "image", "image": f"file://{img}","resized_height": 600, "resized_width": 300} for img in pedestrain_files])
    messages.extend([{"type": "image", "image": f"file://{img}", "resized_height": 300, "resized_width": 150} for img in
                     pedestrain_files])

    messages.append({"type": "text", "text": full_prompt})
    return messages
def extract_conclusion(text):
    # 使用正则表达式匹配 <CONCLUSION> 标签之间的内容
    match = re.search(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "[未找到结论]"
# 主函数
def predict_action_batch(test_image_folder, test_label_folder, test_speed_folder,
                         test_pedestrain_folder, test_boundingbox_folder,
                         question, frame_step, max_images, batch_size=4, max_test_samples=None):
    true_labels= []
    predicted_labels= []

    test_folders = [
        f for f in os.listdir(test_image_folder)
        if os.path.isdir(os.path.join(test_image_folder, f))
        and os.path.exists(os.path.join(test_label_folder, f"{f}.pkl"))
        and os.path.exists(os.path.join(test_speed_folder, f"{f}.pkl"))
    ]
    if max_test_samples:
        test_folders = test_folders[:max_test_samples]

    batch = []
    folder_names = []
    log_lines = []

    for folder_name in tqdm(test_folders):
        image_path = os.path.join(test_image_folder, folder_name)
        label_path = os.path.join(test_label_folder, f"{folder_name}.pkl")
        speed_path = os.path.join(test_speed_folder, f"{folder_name}.pkl")
        bounds_path = os.path.join(test_boundingbox_folder, f"{folder_name}.pkl")
        pedestrain_path = os.path.join(test_pedestrain_folder, folder_name)

        with open(label_path, "rb") as f:
            label = pickle.load(f)
        if label not in [[0], [1]]:
            continue
        gt_label = label[0]

        with open(speed_path, "rb") as f:
            speed = (np.array(pickle.load(f)).squeeze()).tolist()
        with open(bounds_path, "rb") as f:
            ped_box = pickle.load(f)
            ped_box_pixel = convert_normalized_to_pixel_coordinates(ped_box)

        messages = [{"role": "user", "content": create_image_turn(image_path, speed, pedestrain_path, ped_box_pixel, question, frame_step, max_images)}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        batch.append((text, image_inputs, video_inputs, gt_label))
        folder_names.append(folder_name)

        if len(batch) == batch_size or folder_name == test_folders[-1]:
            texts = [b[0] for b in batch]
            images = [b[1] for b in batch]
            videos = [b[2] for b in batch]
            gt_labels = [b[3] for b in batch]

            processor.tokenizer.padding_side = 'left'
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
            print(outputs)

            for i, output in enumerate(outputs):
                output = output.strip()
                output_p=output[0]
                # output_p=extract_conclusion(output)
                pred = 1 if "yes" in output_p else 0 if "no" in output_p else None
                if pred is None:
                    print(f"[⚠️] 无法识别输出: {output}")
                    continue
                print(f"[✅] {folder_names[i]} → 预测: {pred} | 真实: {gt_labels[i]} | 文本: {output}")
                predicted_labels.append(pred)
                true_labels.append(gt_labels[i])
                log_line = f"[✅] {folder_names[i]} → 预测: {pred} | 真实: {gt_labels[i]} | 文本: {output}"
                log_lines.append(log_line)

                # # ✅ 推理输出后显示图像
                # scene_image_file = sorted([
                #     os.path.join(test_image_folder, folder_names[i], img)
                #     for img in os.listdir(os.path.join(test_image_folder, folder_names[i]))
                #     if img.endswith(('.png', '.jpg'))
                # ], key=extract_number)[0]
                # img = mpimg.imread(scene_image_file)
                # plt.figure(figsize=(32, 24))
                # plt.imshow(img)
                # plt.title(f"预测样本: {folder_names[i]} → 预测: {pred} | 真实: {gt_labels[i]}")
                # plt.axis('off')
                # plt.show()
                # input("按回车继续...")

            batch = []
            folder_names = []

    # 评估
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="binary")
    auc = roc_auc_score(true_labels, predicted_labels)
    confusion = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)

    # log_file_path = "../Qwenvl-3B/prediction_log_PIE.txt"
    # with open(log_file_path, "w", encoding="utf-8") as f:
    #     for line in log_lines:
    #         f.write(line + "\n")

    print(classification_report(true_labels, predicted_labels, digits=4))
    print(confusion)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f} | AUC: {auc:.4f} | Accuracy: {accuracy:.4f}")

    with open("../STF/evaluation_result_JAAD-all.txt", "w") as f:
        f.write(f"accuracy:{accuracy:.4f}\n")
        f.write(f"precision:{precision:.4f}\n")
        f.write(f"recall:{recall:.4f}\n")
        f.write(f"f1_score:{f1_score:.4f}\n")
        f.write(f"auc:{auc:.4f}\n")
        for row in confusion:
            f.write(" ".join(map(str, row)) + "\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "auc": auc,
        "accuracy": accuracy,
        "confusion_matrix": confusion
    }

# ⬇️ 主执行入口
if __name__ == "__main__":
    # When using the JAAD dataset, remember to multiply the speed by 10 at line 126.
    # base_path='/JAAD-JAAD_2.0/data/jaad-beh/'
    base_path = "/PIE-master/data/pie"
    # base_path = '/JAAD-JAAD_2.0/data/jaad-all/'
    paths = {
        "train": {
            "image": f"{base_path}/scene image pedestrain bounding box/train",
            "label": f"{base_path}/label/train",
            "speed": f"{base_path}/speed/train",
            "pedestrian_image": f"{base_path}/pedestrian image/train",
            "pedestrian_box": f"{base_path}/pedestrain box/train",
            "cot-label": f"{base_path}/Cot-label/train",
        },
        "test": {
            "image": f"{base_path}/scene image pedestrain bounding box/test",
            "label": f"{base_path}/label/test",
            "speed": f"{base_path}/speed/test",
            "pedestrian_image": f"{base_path}/pedestrian image/test",
            "pedestrian_box": f"{base_path}/pedestrain box/test",
        }
    }

    train_image_path = paths["train"]["image"]
    train_label_path = paths["train"]["label"]
    train_speed_path = paths["train"]["speed"]
    train_pedestrain_image_path = paths["train"]["pedestrian_image"]
    train_pedestrain_box_path = paths["train"]["pedestrian_box"]
    train_cot_label_path = paths["train"]["cot-label"]

    test_image_path = paths["test"]["image"]
    test_label_path = paths["test"]["label"]
    test_speed_path = paths["test"]["speed"]
    test_pedestrain_path = paths["test"]["pedestrian_image"]
    test_boundingbox_path = paths["test"]["pedestrian_box"]


    prompt = ""
    metrics = predict_action_batch(
        test_image_path,
        test_label_path,
        test_speed_path,
        test_pedestrain_path,
        test_boundingbox_path,
        question=prompt,
        frame_step=2,
        max_images=16,
        batch_size=1,
    )
    print(metrics)
