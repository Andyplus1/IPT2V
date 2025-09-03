#!/bin/bash

# Replace with the file path you need.
base_dir="/network_space/server126/shared/hcc/ID_eval_final"

DIMENSION=(
    motion_smoothness
    imaging_quality
)

mp4_dir="$1"
results_dir="$2"

for mp4_file in "$mp4_dir"/*.mp4; do
    video_name=$(basename "$mp4_file" .mp4)
    output_path="$results_dir/$video_name"
    mkdir -p "$output_path"
    python $base_dir/VBench/evaluate.py \
        --dimension "${DIMENSION[@]}" \
        --videos_path "$mp4_file" \
        --mode=custom_input \
        --output_path "$output_path"
done