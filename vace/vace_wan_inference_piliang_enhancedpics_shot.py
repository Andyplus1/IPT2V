# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import time
from datetime import datetime
import logging
import os
import sys
import warnings
import tqdm
import json

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.utils.utils import cache_video, cache_image, str2bool

from models.wan import WanVace
from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from annotators.utils import get_annotator
from PIL import Image
import os

EXAMPLE_PROMPT = {
    "vace-1.3B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    },
    "vace-14B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    }
}



def convert_webp_to_image(webp_path, output_path, output_format='PNG'):
    """
    将WebP文件转换为PNG或JPG格式并保存到指定路径
    
    参数:
        webp_path (str): 输入的WebP文件路径
        output_path (str): 输出文件的路径
        output_format (str): 输出格式，可以是'PNG'或'JPG'（默认为'PNG'）
    
    返回:
        bool: 转换成功返回True，失败返回False
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(webp_path):
            print(f"错误：输入文件 {webp_path} 不存在")
            return False
        
        # 检查输出格式是否有效
        output_format = output_format.upper()
        if output_format not in ['PNG', 'JPG', 'JPEG']:
            print("错误：输出格式必须是'PNG'或'JPG'")
            return False
        
        # 打开WebP文件
        with Image.open(webp_path) as img:
            # 如果是JPG格式且图像有透明度，需要转换为RGB模式
            if output_format in ['JPG', 'JPEG'] and img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # 保存为指定格式
            img.save(output_path, format=output_format)
            print(f"成功将 {webp_path} 转换为 {output_path}")
            return True
            
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        return False



def validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.model_name in WAN_CONFIGS, f"Unsupport model name: {args.model_name}"
    assert args.model_name in EXAMPLE_PROMPT, f"Unsupport model name: {args.model_name}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.model_name], f"Unsupport size {args.size} for model name {args.model_name}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.model_name])}"
    return args


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vace-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The model name to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default='./models/Wan2.1-VACE-14B/',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="视频展示了一位长着尖耳朵的老人，他有一头银白色的长发和小胡子，穿着一件色彩斑斓的长袍，内搭金色衬衫，散发出神秘与智慧的气息。背景为一个华丽宫殿的内部，金碧辉煌。灯光明亮，照亮他脸上的神采奕奕。摄像机旋转动态拍摄，捕捉老人轻松挥手的动作。",
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        default='plain',
        choices=['plain', 'wan_zh', 'wan_en', 'wan_zh_ds', 'wan_en_ds'],
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=2025,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--start_ididx",
        type=int,
        default=0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--end_ididx",
        type=int,
        default=100,
        help="Classifier free guidance scale.")

    return parser


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args
    args = validate_args(args)

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend and args.use_prompt_extend != 'plain':
        prompt_expander = get_annotator(config_type='prompt', config_task=args.use_prompt_extend, return_dict=False)

    cfg = WAN_CONFIGS[args.model_name]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.model_name]["prompt"]
        args.src_video = EXAMPLE_PROMPT[args.model_name].get("src_video", None)
        args.src_mask = EXAMPLE_PROMPT[args.model_name].get("src_mask", None)
        args.src_ref_images = EXAMPLE_PROMPT[args.model_name].get("src_ref_images", None)

    logging.info(f"Input prompt: {args.prompt}")
    if args.use_prompt_extend and args.use_prompt_extend != 'plain':
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt = prompt_expander.forward(args.prompt)
            logging.info(f"Prompt extended from '{args.prompt}' to '{prompt}'")
            input_prompt = [prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    logging.info("Creating WanT2V pipeline.")
    wan_vace = WanVace(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    prompt_json = json.load(open("./prompt_withfacedescription_all_new.json"))
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    sorted_ids = sorted(prompt_json.keys(), key=lambda k: int(k[2:])) 

    # from IPython import embed;embed()
    
    ret_data = {}
    for video in tqdm.tqdm(sorted_ids[args.start_ididx:args.end_ididx]):
        if '.' in video:
            continue
        
        args.src_ref_images = "./data/vip200k_test_track_Facial/"+video+"/image.png"
        src_video, src_mask, src_ref_images = wan_vace.prepare_source([args.src_video],
                                                                      [args.src_mask],
                                                                      [None if args.src_ref_images is None else args.src_ref_images.split(',')],
                                                                      args.frame_num, SIZE_CONFIGS[args.size], device)

        for prompt_num in range(1,6):
            # if '.txt' in i:
            #     prompt_num = i[-5]
            args.prompt = prompt_json[video][str(prompt_num)]
            
            video_name = video+"_"+str(prompt_num)+".mp4"
            
            save_file = "./results/"+video+"_"+str(prompt_num)+".mp4"
            
            print(save_file)
            logging.info(f"Generating video...")
            video_ = wan_vace.generate(
                args.prompt,
                src_video,
                src_mask,
                src_ref_images,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
                negative_prompt = negative_prompt
            )
            
            cache_video(
                tensor=video_[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
            logging.info(f"Saving generated video to {save_file}")
            ret_data['out_video'] = save_file


    
    return ret_data


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
