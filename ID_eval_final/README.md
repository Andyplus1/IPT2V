### Dataset

Identity-Preserving Video Generation (IPVG) task aims to generate videos from textual prompts while maintaining the consistency of the given reference identity throughout the text-to-video generation process. The challenge website is [IPVG](https://hidream-ai.github.io/ipvg-challenge.github.io/). 

Since generating a test pair with the 14B video model takes a considerable amount of time, we sampled 50 unseen IDs and selected one unique prompt for each ID, ensuring that the prompts do not overlap. This resulted in an evaluation dataset with 50 samples, which is sufficient for this task, as validation datasets of a similar scale are also adopted in related works.

### Eval

Here are the evaluation metrics we use for video assessment and the reference code URLs.

- Text Alignment: CLIPScore, GMEScore [gme-Qwen2](https://www.modelscope.cn/models/iic/gme-Qwen2-VL-2B-Instruct)
- Identity Consistency: CurScore, ArcScore [ConsisID](https://github.com/PKU-YuanGroup/ConsisID)
- Video Quality: Motion Smoothness,  Imaging Quality, FID [VBench](https://github.com/Vchitect/VBench)
- OverallScore: 0.3\*GMEScore+0.2\*(CurScore+ArcScore)+0.15\*(Motion Smoothness+Imaging Quality)

### Installation

```
git clone https://github.com/Vchitect/VBench.git
git clone https://github.com/PKU-YuanGroup/ConsisID
```

```bat
conda create -n ipvg_eval python=3.10
conda activate ipvg_eval
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 # or any other PyTorch version with CUDA<=12.1
pip install vbench
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Download the model weights according to the "Download ConsisID" section in [ConsisID ](https://github.com/PKU-YuanGroup/ConsisID), and update the model_path in *cal_face_sim.py* accordingly.

### Usage

We provide a SampleDataset to demonstrate the evaluation code.

- Download SampleDataset [here](https://drive.google.com/file/d/1r3gyArB24D2YaVxcm2iJ7qFAhU38LwGC/view?usp=sharing)

- The three file paths that need to be modified are: base_dir in *run_all.sh* and *run_vbench.sh*， model_path in *cal_face_sim.py*

- ```
  mv cal_face_sim.py ./ConsisID
  mv evaluate.py ./VBench/evaluate.py
  bash run_all.sh
  ```

