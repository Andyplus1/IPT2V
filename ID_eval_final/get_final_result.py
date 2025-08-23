import os
import json
import argparse

def main(results_dir, output_json):
    all_results = []

    for video_folder in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, video_folder)
        if not os.path.isdir(folder_path):
            continue

        result = {"video_name": video_folder}

        # 1. CLIP
        clip_path = os.path.join(folder_path, "GME.json")
        if os.path.exists(clip_path):
            with open(clip_path, "r") as f:
                clip_data = json.load(f)
                result["GME-Score"] = clip_data.get("GME-Score")

        clip_new_path = os.path.join(folder_path, "CLIP.json")
        if os.path.exists(clip_new_path):
            with open(clip_new_path, "r") as f:
                clip_data = json.load(f)
                result["CLIP-Score"] = clip_data.get("CLIP-Score")

        # 2. Face
        face_path = os.path.join(folder_path, "face_similarity.json")
        if os.path.exists(face_path):
            with open(face_path, "r") as f:
                face_data = json.load(f)
                result["cur_score"] = face_data.get("cur_score")
                result["arc_score"] = face_data.get("arc_score")
                result["fid_score"] = face_data.get("fid_score")

        # 4. VBench
        for fname in os.listdir(folder_path):
            if fname.endswith("_Vbench_eval_results.json"):
                with open(os.path.join(folder_path, fname), "r") as f:
                    vbench_data = json.load(f)
                    for k, v in vbench_data.items():
                        if isinstance(v, list) and len(v) > 0:
                            result[k] = v[0]

        all_results.append(result)

    # sum and mean
    video_count = len(all_results)
    indicator_sums = {}
    indicator_counts = {}

    for res in all_results:
        for k, v in res.items():
            if k == "video_name":
                continue
            if isinstance(v, (int, float)):
                indicator_sums[k] = indicator_sums.get(k, 0) + v
                indicator_counts[k] = indicator_counts.get(k, 0) + 1

    indicator_means = {k: (indicator_sums[k] / indicator_counts[k]) for k in indicator_sums}

    final_output = {
        "video_count": video_count,
        "indicator_means": indicator_means,
        "results": all_results
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"All video metrics have been saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results folder")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output json file")
    args = parser.parse_args()
    main(args.results_dir, args.output_json)