# from modelscope import snapshot_download, AutoTokenizer
# from torchgen.api.types import deviceT
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoProcessor
import torch
import os
from torch.utils.data import Dataset
import re
import pickle
from PIL import Image
import numpy as np
import wandb
import accelerate
from transformers import TrainerCallback
import gc
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import random


wandb.init(project="VLMPed-CoT", config={
    "model": "vlmped-cot-stage1",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
})


from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForPermutationLanguageModeling,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer
)

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



model_id = "/data/chl343/VLMPed-CoT-for-Pedestrian-Crossing-Intention-Prediction/LLM-model/Qwen/Qwen2.5-VL-3B-Instruct"



def convert_normalized_to_pixel_coordinates(normalized_bboxes, image_width=1920, image_height=1080):
    """
    将归一化 bounding boxes 转换为图像像素坐标

    参数:
        normalized_bboxes (List[List[float]]): 每帧的归一化框 [x1, y1, x2, y2]，范围在 [0, 1]
        image_width (int): 图像宽度，默认1920
        image_height (int): 图像高度，默认1080

    返回:
        List[List[int]]: 每帧的像素坐标框 [x1, y1, x2, y2]
    """
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


def custom_collate_fn(batch):
    input_ids = pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    labels = pad_sequence(
        [item['labels'] for item in batch],
        batch_first=True,
        padding_value=-100
    )
    # ✅ 正確做法：cat 而不是 stack
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    image_grid_thw = torch.cat([item['image_grid_thw'] for item in batch], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

class IntentDataset(Dataset):
    def __init__(self, image_dir, label_dir,cot_label_dir,speed_dir,pedestrain_dir,pedestrain_box, processor, tokenizer, question,
                 frame_step=3, max_images=16, max_samples=None, max_length=8192,data_set='jaad'):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.question = question
        self.frame_step = frame_step
        self.max_images = max_images
        self.speed_dir = speed_dir
        self.pedestrain_dir = pedestrain_dir
        self.pedestrain_box = pedestrain_box
        self.data_set = data_set
        self.cot_label_dir = cot_label_dir

        def extract_number(filename):
            numbers = re.findall(r'\d+', os.path.basename(filename))
            return int(numbers[0]) if numbers else -1

        folder_list = os.listdir(image_dir)
        if max_samples is not None:
            folder_list = folder_list[:max_samples]

        for folder_name in folder_list:
            image_path = os.path.join(image_dir, folder_name)
            label_path = os.path.join(label_dir, f"{folder_name}.pkl")
            speed_path = os.path.join(speed_dir, f"{folder_name}.pkl")
            ped_box_path = os.path.join(pedestrain_box, f"{folder_name}.pkl")
            cot_label_path = os.path.join(cot_label_dir, f"{folder_name}.pkl")
            pedestrain_path = os.path.join(pedestrain_dir, folder_name)


            if not os.path.exists(label_path) or not os.path.exists(speed_path) or not os.path.exists(ped_box_path) or not os.path.exists(cot_label_path):
                continue

            with open(label_path, "rb") as f:
                label = pickle.load(f)
            if label[0] not in [0, 1]:
                continue

            with open(speed_path, "rb") as f:
                speed = pickle.load(f)
            if self.data_set=="jaad":
                speed = (np.array(speed).squeeze()*10).tolist()
            elif self.data_set=="pie":
                speed = (np.array(speed).squeeze()).tolist()
            else:
                print('数据集有误！')

            with open(cot_label_path, "rb") as f:
                cot_label = pickle.load(f)


            with open(ped_box_path, "rb") as f:
                ped_box = pickle.load(f)
            ped_box_pixel= convert_normalized_to_pixel_coordinates(ped_box)


            answer = "yes" if label[0] == 1 else "no"

            image_files = sorted([
                os.path.join(image_path, img)
                for img in os.listdir(image_path)
                if img.endswith((".png", ".jpg", ".jpeg"))
            ], key=extract_number)

            pedestrain_files = sorted([
                os.path.join(pedestrain_path, img)
                for img in os.listdir(pedestrain_path)
                if img.endswith((".png", ".jpg", ".jpeg"))
            ], key=extract_number)

            image_files = image_files[::frame_step][:max_images]
            pedestrain_files = pedestrain_files[::frame_step][:max_images]

            # ✅ 速度采样方式与图像一致
            sampled_speed = speed[::frame_step][:max_images]
            ped_box_pixel = ped_box_pixel[::frame_step][:max_images]

            self.data.append({
                "images": image_files,
                "question": question,
                "answer": answer,
                "cot_answer": cot_label,
                "speed": sampled_speed,
                "pedestrain":pedestrain_files,
                "bounding_box": ped_box_pixel,

            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        prompt_text = (
            f"You are an intelligent vision model. Your task is to determine whether a pedestrian has the intention to cross the street within the next 1–2 seconds, based on a sequence of images (approximately 0.5 seconds) captured by an in-vehicle camera and a textual description."
            f"The first image is a scene image captured by the in-vehicle camera, with the target pedestrian highlighted by a red bounding box. The remaining images are a sequence showing the pedestrian's continuous poses. Below is the sequence of bounding box coordinates for the target pedestrian in each frame, in the format [x1, y1, x2, y2], where (x1, y1) represents the top-left corner and (x2, y2) the bottom-right corner: {example['bounding_box']}.The following is the sequence of vehicle speeds corresponding to each frame (unit: km/h): {example['speed']}."
            f"Please reason step-by-step using all of the above information to infer whether the pedestrian has the intention to cross the street within the next 1–2 seconds. Your reasoning process must strictly follow the format composed of the following four specific sections: SUMMARY, CAPTION, REASONING, and CONCLUSION. Use the following format for your reasoning process : <SUMMARY>[Briefly summarize how you will solve the problem and the steps you will take to reach the answer.]</SUMMARY> <CAPTION>[Describe the image contents in detail, emphasizing parts relevant to the question.]</CAPTION> <REASONING>[Step-by-step explanation of your logical reasoning process.]</REASONING> <CONCLUSION>[Directly give the final answer, must match the correct answer exactly, directly output the answer 'Yes' or 'No']</CONCLUSION>."
        )

        full_prompt = prompt_text  + example["question"]
        messages = [{"type": "image", "image": f"file://{img}"}
                    for img in [example["images"][0]]]

        messages.extend([
            {"type": "image", "image": f"file://{img}","resized_height": 600, "resized_width": 300}
            for img in example["pedestrain"]
        ])

        messages.append({"type": "text", "text": full_prompt})
        messages = [{"role": "user", "content": messages}]
        # output_content = example["answer"]
        output_content = example["cot_answer"]


        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device=next(model.parameters()).device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        inputs = {key: value.tolist() for key, value in inputs.items()}
        instruction = inputs

        response = self.tokenizer(f"{output_content}", add_special_tokens=False)

        input_ids = instruction["input_ids"][0] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"][0] + [1] * (len(response["input_ids"]) + 1)
        labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        labels = [-100 if t == self.tokenizer.pad_token_id else t for t in labels]

        # debug
        # grid = torch.tensor(inputs['image_grid_thw'])
        # print(f"idx={idx}, grid_thw raw shape: {grid.shape}, n_images={len(image_inputs)}")
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "pixel_values": torch.tensor(inputs['pixel_values']),
            "image_grid_thw": torch.tensor(inputs['image_grid_thw']).squeeze(0),
        }


# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)

# model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto",
#                                                         torch_dtype=torch.bfloat16, trust_remote_code=True, )
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map='auto', trust_remote_code=True, )
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

question = ""


base_path_pie = "/data/chl343/VLMPed-CoT-for-Pedestrian-Crossing-Intention-Prediction/pedestrian data generation/data/pie"
paths_pie = {
    "train": {
        "image": f"{base_path_pie}/scene image pedestrain bounding box/train",
        "label": f"{base_path_pie}/label/train",
        "cot_label":f"{base_path_pie}/Cot_label/train",
        "speed": f"{base_path_pie}/speed/train",
        "pedestrian_image": f"{base_path_pie}/pedestrian image/train",
        "pedestrian_box": f"{base_path_pie}/pedestrain box/train",

    },
    "test": {
        "image": f"{base_path_pie}/scene image pedestrain bounding box/test",
        "label": f"{base_path_pie}/label/test",
        "speed": f"{base_path_pie}/speed/test",
        "pedestrian_image": f"{base_path_pie}/pedestrian image/test",
        "pedestrian_box": f"{base_path_pie}/pedestrain box/test",
    }
}

train_image_path_pie = paths_pie["train"]["image"]
train_label_path_pie = paths_pie["train"]["label"]
train_cot_label_path_pie = paths_pie["train"]["cot_label"]
train_speed_path_pie = paths_pie["train"]["speed"]
train_pedestrain_image_path_pie = paths_pie["train"]["pedestrian_image"]
train_pedestrain_box_path_pie = paths_pie["train"]["pedestrian_box"]
train_dataset_pie = IntentDataset(
    train_image_path_pie,
    train_label_path_pie,
    train_cot_label_path_pie,
    train_speed_path_pie,
    train_pedestrain_image_path_pie,
    train_pedestrain_box_path_pie,
    processor,
    tokenizer,
    question,
    frame_step=2,
    max_images=16,
    max_length=8192,
    data_set="pie")


base_path_jaad = "/data/chl343/VLMPed-CoT-for-Pedestrian-Crossing-Intention-Prediction/pedestrian data generation/data/jaad-all"
paths_jaad = {
    "train": {
        "image": f"{base_path_jaad}/scene image pedestrain bounding box/train",
        "label": f"{base_path_jaad}/label/train",
        "cot_label":f"{base_path_jaad}/Cot_label/train",
        "speed": f"{base_path_jaad}/speed/train",
        "pedestrian_image": f"{base_path_jaad}/pedestrian image/train",
        "pedestrian_box": f"{base_path_jaad}/pedestrain box/train",
    }
}

train_image_path_jaad = paths_jaad["train"]["image"]
train_label_path_jaad = paths_jaad["train"]["label"]
train_cot_label_path_jaad = paths_jaad["train"]["cot_label"]
train_speed_path_jaad = paths_jaad["train"]["speed"]
train_pedestrain_image_path_jaad = paths_jaad["train"]["pedestrian_image"]
train_pedestrain_box_path_jaad = paths_jaad["train"]["pedestrian_box"]

train_dataset_jaad = IntentDataset(
    train_image_path_jaad,
    train_label_path_jaad,
    train_cot_label_path_jaad,
    train_speed_path_jaad,
    train_pedestrain_image_path_jaad,
    train_pedestrain_box_path_jaad,
    processor,
    tokenizer,
    question,
    frame_step=2,
    max_images=16,
    max_length=8192,
    data_set='jaad'
)

train_dataset = ConcatDataset([train_dataset_pie, train_dataset_jaad])



# 获取 ConcatDataset 的长度
total_len = len(train_dataset)

# 生成乱序索引
indices = list(range(total_len))
random.shuffle(indices)

# 用 Subset 包装打乱的索引
shuffled_train_dataset = Subset(train_dataset, indices)


class MemoryCleanCallback(TrainerCallback):
    def __init__(self, interval: int = 10):
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0:
            print(f"[Step {state.global_step}] 清理显存中...")
            torch.cuda.empty_cache()
            gc.collect()


# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=16,  # Lora 秩
    lora_alpha=32,  # Lora alaph
    lora_dropout=0.1,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
peft_model = get_peft_model(model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir="./output/",
    per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", 2)),
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=500,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    bf16=True,
    report_to="wandb",
    seed=42,
    dataloader_num_workers=int(os.getenv("DATALOADER_NUM_WORKERS", 0)),
)


# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=shuffled_train_dataset,
    # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    data_collator=custom_collate_fn,
    callbacks=[MemoryCleanCallback(interval=5)],
)

trainer.train()