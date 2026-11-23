import os
import json
import glob
import argparse
import pandas as pd

DEFAULT_RESULTS_DIR = "/mnt/parscratch/users/acd22dg/experiments/results"
DEFAULT_OUT_DIR = "/mnt/parscratch/users/acd22dg/experiments"


def load_results(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                print(f"WARNING: could not parse {path}, skipping (job may still be writing)")
                continue
        rows.append(content)
    if not rows:
        raise SystemExit(f"No result files found in {results_dir}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    df = load_results(args.results_dir)
    df["wer"] = df["fine_tuned_wer"].combine_first(df["zero_shot_wer"])

    os.makedirs(args.out_dir, exist_ok=True)

    long_path = os.path.join(args.out_dir, "master_results_long.csv")
    df.sort_values(["config", "speaker"]).to_csv(long_path, index=False)
    print(f"Wrote long-format table -> {long_path} ({len(df)} rows)")

    pivot = df.pivot_table(index="config", columns="speaker", values="wer")

    wide_path = os.path.join(args.out_dir, "master_wer_summary.csv")
    pivot.to_csv(wide_path)
    print(f"Wrote wide/pivot table -> {wide_path}")

    print(f"\nCollected {len(df)} result files from {args.results_dir}")
    print("\n" + "=" * 80)
    print(" WER SUMMARY (rows = config, cols = speaker) ")
    print("=" * 80)
    print(pivot.to_string())


if __name__ == "__main__":
    main()