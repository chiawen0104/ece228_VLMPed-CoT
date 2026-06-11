# ECE228 VLMPed-CoT

Final Project of UCSD ECE 228. Our project title: **How Do Vision Language Models Utilize Multi-Frame
Temporal Information for Pedestrian Intention Prediction?**

We design controlled experiments to analyze how VLMPed-CoT and VLMPed-wo-CoT utilize multi-frame temporal information for pedestrian crossing intention prediction.  

This repository is adapted from the official implementation of:

**VLMPed-CoT: A Large Vision-Language Model with Chain-of-Thought Mechanism for Pedestrian Crossing Intention Prediction**

Original repository: [lyc2121/VLMPed-CoT-for-Pedestrian-Crossing-Intention-Prediction](https://github.com/lyc2121/VLMPed-CoT-for-Pedestrian-Crossing-Intention-Prediction)  
Original paper: https://www.sciopen.com/article/10.26599/COMMTR.2026.9640009


Our fine-tuned model weights are available on Hugging Face:

| Model | Link |
|-------|------|
| VLMPed-CoT | [chiawen0104/VLMPed-CoT](https://huggingface.co/chiawen0104/VLMPed-CoT) |
| VLMPed-wo-CoT | [chiawen0104/VLMPed-wo-CoT](https://huggingface.co/chiawen0104/VLMPed-wo-CoT) |

---

## Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

You will also need to register and configure:

- **Google Gemini API key** — for running CoT data generation scripts under `gemini/`

---

## Environment Setup

### 1. HRNet Weights (Pose Estimation)

Download the HRNet model weights from:

- [pose_hrnet_w48_384x288.pth](https://drive.google.com/open?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS)

Place the weight file into:

```bash
pedestrian_data_generation/hrnet/weights/pose_hrnet_w48_384x288.pth
```
For more information, please refer to [simple-HRNet](https://github.com/stefanopini/simple-HRNet).

### 2. Qwen2.5-VL-3B-Instruct Model Weights

Download the base model from Hugging Face:
use the `huggingface_hub` Python package:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    local_dir="LLM-model/Qwen/Qwen2.5-VL-3B-Instruct"
)
```

## Data Preparation

Download the datasets:

- **JAAD**: https://github.com/ykotseruba/JAAD
- **PIE**: https://github.com/aras62/PIE

Then run the scripts under `pedestrian_data_generation/` to process the data:

1. Extract frames from videos:
```bash
   python pedestrian_data_generation/extract_image.py
```

2. Generate training and testing data:
```bash
   python pedestrian_data_generation/Data_generation_jaad.py
   python pedestrian_data_generation/Data_generation_pie.py
```

## CoT Data Generation
Run the following scripts under `gemini/` to generate Chain-of-Thought data:

```bash
python gemini/gemini_cot_data_generation_JAAD.py
python gemini/gemini_cot_data_generation_PIE.py
```

Configure your Gemini API key before running.

## Model Fine-tuning

Fine-tune the Qwen model in two stages:

```bash
# Stage 1
bash qwen/run_stage1.sh

# Stage 2
bash qwen/run_stage2.sh
```

Or run the Python scripts directly:

```bash
python qwen/fine_tuning_COT-joint_stage_1.py
python qwen/fine_tuning_COT-joint_stage_2.py
```

## Model Testing

Evaluate the fine-tuned model:

```bash
python qwen/test_fine_tuning_cot_joint.py
```

## Final Data Structure
```
├── gemini/                          # CoT data generation & inference scripts
│   ├── gemini_cot_data_generation_JAAD.py
│   ├── gemini_cot_data_generation_PIE.py
│   ├── gemini_cot_inference.py
│   └── gemini_without_cot_inference.py
│
├── pedestrian_data_generation/      # Data preprocessing pipeline
│   ├── data/
│   │   ├── jaad-all/
│   │   │   ├── Cot_label/
│   │   │   ├── label/
│   │   │   ├── pdestrain pose/
│   │   │   ├── pedestrain box/
│   │   │   ├── pedestrian image/
│   │   │   └── scene image pedestrain bounding box/
│   │   └── pie/
│   ├── data_parameter/
│   ├── hrnet/
│   │   └── weights/                 # Place pose_hrnet_w48_384x288.pth here
│   ├── data_generation_class.py
│   ├── Data_generation_jaad.py
│   └── Data_generation_pie.py
│   └── extract_image.py
│   └── jaad_data.py
│   └── pie_data.py
│
├── qwen/                            # Model fine-tuning & evaluation
│   ├── experiments/
│   │   └── results/                 # Experiment Results 
│   │   ├── exp1_multi_effect.py
│   │   ├── exp2a_frame_ablation.py
│   │   ├── exp2b_frame_duplication.py
│   │   └── exp3_temporal_sensitivity.py
│   ├── fine_tuning_COT-joint_stage_1.py
│   ├── fine_tuning_COT-joint_stage_2.py
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   ├── test_fine_tuning_cot_joint.py
│   ├── test_cot_joint_without_fine-tuning.py
│   └── test_without_cot_inference.py
│
└── LLM-model/
    └── Qwen/
        └── Qwen2.5-VL-3B-Instruct/  # Place downloaded model here
```

