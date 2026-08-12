import os
import json
import time
import argparse
import traceback
from dataclasses import dataclass
from typing import Optional

from training_pipeline import WhisperTrainingPipeline

# ============================================================
# CONFIG
# ============================================================

SPEAKERS = ["F000", "M001", "F002", "M003", "M004", "M005"]

RESULTS_DIR = "/mnt/parscratch/users/acd22dg/experiments/results"


@dataclass
class IntermediateConfig:
    name: str
    path: str
    sample_rate: int = 16000
    epochs: int = 3

@dataclass
class FinalTuneConfig:
    name: str
    path: str
    epochs: int = 10

@dataclass
class TrainingTask:
    name: str
    finetune: Optional[FinalTuneConfig] = None
    intermediate: Optional[IntermediateConfig] = None
    speakers: Optional[list] = None

# ============================================================
# STAGES
# ============================================================

inters = [
    None,
    IntermediateConfig(
        name="trans_tts",
        path="/mnt/parscratch/users/acd22dg/datasets-final/transformertts",
        sample_rate=16000,
        epochs=5
    ),
    IntermediateConfig(
        name="nactts_general",
        path="/mnt/parscratch/users/acd22dg/datasets-final/nac-all",
        sample_rate=16000,
        epochs=5
    ),
    IntermediateConfig(
        name="nactts_speaker",
        path="/mnt/parscratch/users/acd22dg/datasets-final/nac-personal",
        sample_rate=16000,
        epochs=5
    ),
]

finals = [
    None,
    FinalTuneConfig(
        name="all_speakers",
        path="/mnt/parscratch/users/acd22dg/datasets-final/all",
        epochs=10
    ),
    FinalTuneConfig(
        name="target_speaker",
        path="/mnt/parscratch/users/acd22dg/datasets-final/target",
        epochs=10
    ),
]

configs: list[TrainingTask] = []

for inter in inters:
    for final in finals:
        inter_str = inter.name if inter else "no_intermediate"
        final_str = final.name if final else "no_finetune"
        task_name = f"{inter_str}__vs__{final_str}"

        configs.append(TrainingTask(
            name=task_name,
            intermediate=inter,
            finetune=final,
            speakers=SPEAKERS
        ))

NUM_CONFIGS = len(configs)
NUM_SPEAKERS = len(SPEAKERS)
# Total SLURM array size needed to cover every (config, speaker) pair:
#   --array=0-{NUM_CONFIGS * NUM_SPEAKERS - 1}

# ============================================================
# WORKER (single config, single speaker, single process)
# ============================================================

def run_task_for_speaker(task: TrainingTask, speaker: str) -> dict:
    """Run exactly one speaker through exactly one experiment config.

    Mirrors the branching that used to live in run_task(), but scoped to a
    single speaker so each SLURM array task gets a completely fresh process
    (fresh CUDA context, fresh Trainer state) - no more same-process
    multi-speaker looping.
    """
    base_output_dir = f"/mnt/parscratch/users/acd22dg/experiments/{task.name}"

    inter_dir = task.intermediate.path if task.intermediate else None
    inter_meta = os.path.join(inter_dir, "metadata.csv") if inter_dir else None

    if task.finetune:
        fine_dir = task.finetune.path
        fine_meta = os.path.join(fine_dir, "metadata.csv")
        fine_epochs = task.finetune.epochs
    else:
        fine_dir = "/mnt/parscratch/users/acd22dg/datasets-final/target"
        fine_meta = os.path.join(fine_dir, "metadata.csv")
        fine_epochs = 0

    pipeline = WhisperTrainingPipeline(
        sample_rate=task.intermediate.sample_rate if task.intermediate else 16000,
        intermediate_dataset_dir=inter_dir,
        intermediate_metadata_path=inter_meta,
        intermediate_output_dir=os.path.join(base_output_dir, "intermediate_ckpt"),
        intermediate_num_train_epochs=task.intermediate.epochs if task.intermediate else 30,
        finetune_dataset_path=fine_dir,
        finetune_metadata_path=fine_meta,
        finetune_output_dir=os.path.join(base_output_dir, "finetune_results"),
        finetune_num_train_epochs=fine_epochs,
        speakers=[speaker],
    )

    print(f"############ CONFIG: {task.name} | SPEAKER: {speaker} | "
          f"INTERMEDIATE: {task.intermediate.name if task.intermediate else 'None'} | "
          f"FINE-TUNING: {task.finetune.name if task.finetune else 'None'} ############")

    result = {
        "config": task.name,
        "intermediate": task.intermediate.name if task.intermediate else None,
        "finetune": task.finetune.name if task.finetune else None,
        "speaker": speaker,
        "zero_shot_wer": None,
        "fine_tuned_wer": None,
    }

    if task.intermediate and task.finetune:
        speaker_data = pipeline.load_speaker_data(pipeline.finetune_dataset_path)
        train_ds, eval_ds = pipeline.prepare_finetune_dataset(speaker, speaker_data)

        model_dir = pipeline.run_intermediate(target_speaker=speaker)
        wer = pipeline._finetune_single_speaker(speaker, speaker_data, model_dir, train_ds=train_ds, eval_ds=eval_ds)
        result["fine_tuned_wer"] = wer

    elif task.intermediate and not task.finetune:
        model_dir = pipeline.run_intermediate(target_speaker=speaker)
        table = pipeline.run_inference(base_model_dir=model_dir, target_speakers=[speaker])
        if speaker in table.index:
            result["zero_shot_wer"] = float(table.loc[speaker, "zero_shot"])

    elif not task.intermediate and task.finetune:
        speaker_data = pipeline.load_speaker_data(pipeline.finetune_dataset_path)
        wer = pipeline._finetune_single_speaker(speaker, speaker_data, pipeline.model_id)
        result["fine_tuned_wer"] = wer

    else:
        table = pipeline.run_inference(base_model_dir=pipeline.model_id, target_speakers=[speaker])
        if speaker in table.index:
            result["zero_shot_wer"] = float(table.loc[speaker, "zero_shot"])

    return result


def write_result(result: dict):
    """Write this job's result as its own JSON file so concurrently running
    array tasks never contend for the same file. collect_results.py later
    globs this directory and merges everything into one table."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = f"{result['config']}__{result['speaker']}.json"
    out_path = os.path.join(RESULTS_DIR, fname)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, out_path)
    print(f"Wrote result -> {out_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--Config", required=True, type=int, help="Index into configs[]")
    parser.add_argument("-s", "--Speaker", required=True, help="Speaker ID, e.g. F000")
    args = parser.parse_args()

    print(f"Configuration used: {args.Config} | Speaker: {args.Speaker}")
    task = configs[args.Config]

    result = {
        "config": task.name,
        "intermediate": task.intermediate.name if task.intermediate else None,
        "finetune": task.finetune.name if task.finetune else None,
        "speaker": args.Speaker,
        "zero_shot_wer": None,
        "fine_tuned_wer": None,
        "status": "ok",
        "error": None,
        "runtime_seconds": None,
    }

    start = time.time()
    try:
        result.update(run_task_for_speaker(task, args.Speaker))
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        result["runtime_seconds"] = round(time.time() - start, 1)
        write_result(result)

    if result["status"] == "failed":
        raise SystemExit(1)

    return result


if __name__ == "__main__":
    main()