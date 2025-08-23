#!/bin/bash

# Replace with the file path you need.
base_dir="/network_space/server126/shared/hcc/ID_eval_final"

mp4_dir="$base_dir/SampleDataset/vace_stg"
results_dir="$base_dir/SampleDataset/Results_vace_stg"
prompts_dir="$base_dir/SampleDataset/prompts"
ref_image_dir="$base_dir/SampleDataset/ID_image"  


# 1. Calculate CLIPScore and GMEScore

python $base_dir/CLIPCal.py \
    "$mp4_dir" \
    "$prompts_dir" \
    "$results_dir"

python $base_dir/GMECal.py \
    "$mp4_dir" \
    "$prompts_dir" \
    "$results_dir"


# 2. Calculate Vbench metrics

bash $base_dir/run_vbench.sh "$mp4_dir" "$results_dir"

# 3. Calculate face identity similarity

for mp4_file in "$mp4_dir"/*.mp4; do
    video_name=$(basename "$mp4_file" .mp4)
    out_dir="$results_dir/$video_name"
    mkdir -p "$out_dir"
    # Extract prefix as reference image name (e.g. id001_prompt2.mp4 -> id001.png)
    ref_prefix=${video_name%%_*}
    ref_image="$ref_image_dir/${ref_prefix}.png"
    if [ ! -f "$ref_image" ]; then
        echo "Reference image $ref_image does not exist, skip $video_name"
        continue
    fi
    python "$base_dir/ConsisID/cal_face_sim.py" \
        "$mp4_file" "$ref_image" "$out_dir/face_similarity.json"
done


python $base_dir/get_final_result.py \
    --results_dir "$results_dir" \
    --output_json "$results_dir/final_results.json"

cp "$0" "$results_dir/"