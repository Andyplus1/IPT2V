import torch
import torch.nn.functional as F
import av
import numpy as np
from transformers import  AutoModel
from PIL import Image
from typing import List
import os
import sys
import json

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load GME model
print("Loading GME model...")
gme = AutoModel.from_pretrained(
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    torch_dtype="float16", 
    device_map='cuda',
    trust_remote_code=True
)
print("GME model loaded.")

def extract_frames(video_path: str) -> List[Image.Image]:
    frames = []
    try:
        print(f"Opening video {video_path} to extract ALL frames...")
        container = av.open(video_path)
        stream = container.streams.video[0]
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
        container.close()
        print(f"Successfully extracted {len(frames)} frames.")
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return []
    if len(frames) > 300:
        print(f"Warning: Extracted {len(frames)} frames. This might consume a lot of memory and take a long time to process.")
    return frames

def sample_frames(frames, num_samples=16):
    """Uniformly sample num_samples frames"""
    if len(frames) <= num_samples:
        return frames
    idxs = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    return [frames[i] for i in idxs]

def calculate_gme_similarity(gme, video_path, prompt, batch_size=4, num_samples=16):
    """After sampling, calculate the GME similarity between each video frame and the prompt"""
    instruction = "Find frames that match the given text description."
    text_embedding = gme.get_text_embeddings(
        texts=[prompt], 
        instruction=instruction
    )
    print(f"Extracting video frames: {video_path}")
    frames = extract_frames(video_path)
    print(f"Total extracted {len(frames)} frames")
    frames = sample_frames(frames, num_samples=num_samples)
    print(f"Sampled {len(frames)} frames for GME calculation")
    similarities = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        # Directly pass PIL.Image objects, no need to save as temp files
        image_embeddings = gme.get_image_embeddings(
            images=batch_frames,
            is_query=False
        )
        batch_similarities = (text_embedding @ image_embeddings.T).tolist()[0]
        similarities.extend(batch_similarities)
        if (i + batch_size) % 100 == 0:
            print(f"Processed {i + batch_size}/{len(frames)} frames")
    avg_similarity = float(np.mean(similarities))
    return avg_similarity, similarities

# --- Batch processing ---
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python GMECal.py <mp4_folder> <prompts_folder> <results_dir>")
        sys.exit(1)
    mp4_folder = sys.argv[1]
    prompts_folder = sys.argv[2]
    results_dir = sys.argv[3]

    for video_file in sorted(os.listdir(mp4_folder)):
        if not video_file.endswith(".mp4"):
            continue
        video_path = os.path.join(mp4_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        prompt_path = os.path.join(prompts_folder, f"{video_name}.txt")
        if not os.path.exists(prompt_path):
            print(f"Prompt file not found for {video_file}, expected: {prompt_path}")
            continue
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            description = f.read().strip()
        print(f"\nProcessing: {video_file}")
        print(f"Prompt: {description}")
        
        
        gme_avg, gme_all = calculate_gme_similarity(gme, video_path, description)
        
        print(f"\n=====================================")
        print(f"Video: {video_file}")
        print(f"Prompt: '{description}'")
        print(f"-------------------------------------")
        
        print(f"GME Avg Similarity: {gme_avg:.4f}")
        print(f"=====================================")
            
        # Save as json
        out_dir = os.path.join(results_dir, video_name)
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, "GME.json")
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump({
                "video": video_file,
                "prompt": description,
                "GME-Score": gme_avg,
                "GME_frame_similarities": gme_all
            }, jf, ensure_ascii=False, indent=2)