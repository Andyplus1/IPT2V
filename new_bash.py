#!/usr/bin/env python3

import os

base_command = """CUDA_VISIBLE_DEVICES={gpu_id} python vace/vace_wan_inference_piliang_enhancedpics_shot.py \\
--src_ref_images "/root/autodl-tmp/VACE-Benchmark/assets/examples/face/src_ref_image_1.png" \\
--prompt "视频展示了一位长着尖耳朵的老人，他有一头银白色的长发和小胡子，穿着一件色彩斑斓的长袍，内搭金色衬衫，散发出神秘与智慧的气息。背景为一个华丽宫殿的内部，金碧辉煌。灯光明亮，照亮他脸上的神采奕奕。摄像机旋转动态拍摄，捕捉老人轻松挥手的动作。" \\
--size 832*480 \\
--start_ididx {start_idx} \\
--end_ididx {end_idx}"""

for i in range(20):
    gpu_id = i % 5
    start_idx = i * 10
    end_idx = start_idx + 10
    
    content = f"""#!/bin/bash
{base_command.format(gpu_id=gpu_id, start_idx=start_idx, end_idx=end_idx)}
"""
    
    filename = f"run{i}.sh"
    with open(filename, 'w') as f:
        f.write(content)
    
    os.chmod(filename, 0o755)
    print(f"Generated {filename}")