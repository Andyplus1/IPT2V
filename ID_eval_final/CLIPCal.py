import torch
import torch.nn.functional as F
import av
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List
import os
import sys
import json

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load CLIP model
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
print("CLIP model loaded.")

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

def sample_frames(frames, num_samples=32):
    """Uniformly sample num_samples frames"""
    if len(frames) <= num_samples:
        return frames
    idxs = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    return [frames[i] for i in idxs]

def calculate_clip_similarity(clip_model, clip_processor, video_path, prompt, batch_size=32, num_samples=32):
    print(f"Extracting video frames: {video_path}")
    frames = extract_frames(video_path)
    print(f"Total extracted {len(frames)} frames")
    frames = sample_frames(frames, num_samples=num_samples)
    print(f"Sampled {len(frames)} frames for CLIP calculation")
    similarities = []
    # Text encoding, force truncation
    text_inputs = clip_processor(
        text=[prompt], 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=77
    ).to(device)
    # Check if truncated
    tokenized = clip_processor.tokenizer.encode(prompt)
    truncated = len(tokenized) > 77
    # Text features
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)
    # Image encoding
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        inputs = clip_processor(images=batch_frames, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
        # Calculate cosine similarity
        sim = (image_features @ text_features.T).squeeze(-1).cpu().numpy()
        similarities.extend(sim.tolist())
        if (i + batch_size) % 100 == 0:
            print(f"Processed {i + batch_size}/{len(frames)} frames")
    avg_similarity = float(np.mean(similarities))
    return avg_similarity, similarities, truncated

# --- Batch processing ---
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python CLIP.py <mp4_folder> <prompts_folder> <results_dir>")
        sys.exit(1)
    mp4_folder = sys.argv[1]
    prompts_folder = sys.argv[2]
    results_dir = sys.argv[3]

    truncated_count = 0
    total_count = 0

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
        
        clip_avg, clip_all, truncated = calculate_clip_similarity(clip_model, clip_processor, video_path, description)
        if truncated:
            truncated_count += 1
        total_count += 1

        print(f"\n=====================================")
        print(f"Video: {video_file}")
        print(f"Prompt: '{description}'")
        print(f"-------------------------------------")
        print(f"CLIP Avg Similarity: {clip_avg:.4f}")
        print(f"=====================================")
            
        # Save as json
        out_dir = os.path.join(results_dir, video_name)
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, "CLIP.json")
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump({
                "video": video_file,
                "prompt": description,
                "CLIP-Score": clip_avg,
                "CLIP_frame_similarities": clip_all
            }, jf, ensure_ascii=False, indent=2)

    print(f"\nA total of {truncated_count} texts were truncated during processing, accounting for {truncated_count/total_count:.2%}")