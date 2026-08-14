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

DEFAULT_WHISPER_VERSION = "openai/whisper-medium"

FREEZE_ENCODER = False
FREEZE_DECODER = False
INTERMEDIATE_LEARNING_RATE = 1e-5
FINETUNE_LEARNING_RATE = 1e-5

INTERMEDIATE_PER_DEVICE_BATCH_SIZE = 16
INTERMEDIATE_GRAD_ACCUM_STEPS = 4
FINETUNE_PER_DEVICE_BATCH_SIZE = 8
FINETUNE_GRAD_ACCUM_STEPS = 8

INTERMEDIATE_EPOCHS = 5
FINETUNE_EPOCHS = 10


@dataclass
class IntermediateConfig:
    name: str
    path: str
    sample_rate: int = 16000
    epochs: int = INTERMEDIATE_EPOCHS

@dataclass
class FinalTuneConfig:
    name: str
    path: str
    epochs: int = FINETUNE_EPOCHS

@dataclass
class TrainingTask:
    name: str
    finetune: Optional[FinalTuneConfig] = None
    intermediate: Optional[IntermediateConfig] = None
    speakers: Optional[list] = None

# ============================================================
# STAGES
# ============================================================

INTERMEDIATES = {
    "no_inter": None,
    "trans_tts": IntermediateConfig(
        name="trans_tts",
        path="/mnt/parscratch/users/acd22dg/datasets-final/transformertts",
        sample_rate=16000,
    ),
    "nactts_gen": IntermediateConfig(
        name="nactts_gen",
        path="/mnt/parscratch/users/acd22dg/datasets-final/nac-all",
        sample_rate=16000,
    ),
    "nactts_spk": IntermediateConfig(
        name="nactts_spk",
        path="/mnt/parscratch/users/acd22dg/datasets-final/nac-personal",
        sample_rate=16000,
    ),
}

FINETUNES = {
    "no_ft": None,
    "all": FinalTuneConfig(
        name="all",
        path="/mnt/parscratch/users/acd22dg/datasets-final/all-new",
    ),
    "target": FinalTuneConfig(
        name="target",
        path="/mnt/parscratch/users/acd22dg/datasets-final/target-new",
    ),
}

configs: list[TrainingTask] = [
    TrainingTask(name=f"medium_{inter_key}__{ft_key}",
                 intermediate=INTERMEDIATES[inter_key],
                 finetune=FINETUNES[ft_key],
                 speakers=SPEAKERS)
    for inter_key in ["no_inter", "trans_tts", "nactts_gen", "nactts_spk"]
    for ft_key in ["no_ft", "all", "target"]
]

NUM_CONFIGS = len(configs)
NUM_SPEAKERS = len(SPEAKERS)
# Total SLURM array size needed to cover every (config, speaker) pair:
#   --array=0-{NUM_CONFIGS * NUM_SPEAKERS - 1}   (12 * 6 - 1 = 71)

# ============================================================
# WORKER (single config, single speaker, single process)
# ============================================================

def run_task_for_speaker(task: TrainingTask, speaker: str, whisper_version: str = DEFAULT_WHISPER_VERSION) -> dict:

    version_tag = whisper_version.replace("/", "_")
    base_output_dir = f"/mnt/parscratch/users/acd22dg/experiments/{task.name}/{version_tag}"

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
        model_id=whisper_version,
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
        freeze_encoder=FREEZE_ENCODER,
        freeze_decoder=FREEZE_DECODER,
        intermediate_learning_rate=INTERMEDIATE_LEARNING_RATE,
        finetune_learning_rate=FINETUNE_LEARNING_RATE,
        intermediate_per_device_batch_size=INTERMEDIATE_PER_DEVICE_BATCH_SIZE,
        intermediate_grad_accum_steps=INTERMEDIATE_GRAD_ACCUM_STEPS,
        finetune_per_device_batch_size=FINETUNE_PER_DEVICE_BATCH_SIZE,
        finetune_grad_accum_steps=FINETUNE_GRAD_ACCUM_STEPS,
    )

    print(f"############ CONFIG: {task.name} | SPEAKER: {speaker} | "
          f"WHISPER: {whisper_version} | "
          f"INTERMEDIATE: {task.intermediate.name if task.intermediate else 'None'} | "
          f"FINE-TUNING: {task.finetune.name if task.finetune else 'None'} | "
          f"freeze_encoder={FREEZE_ENCODER} freeze_decoder={FREEZE_DECODER} ############")

    result = {
        "config": task.name,
        "whisper_version": whisper_version,
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
    version_tag = result.get("whisper_version", DEFAULT_WHISPER_VERSION).replace("/", "_")
    fname = f"{result['config']}__{result['speaker']}__{version_tag}.json"
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
    parser.add_argument(
        "-w", "--WhisperVersion",
        default=DEFAULT_WHISPER_VERSION,
        help=(
            "Pretrained Whisper checkpoint to start from, e.g. "
            "openai/whisper-small, openai/whisper-medium, openai/whisper-large-v3. "
            f"Default: {DEFAULT_WHISPER_VERSION}"
        ),
    )
    args = parser.parse_args()

    print(f"Configuration used: {args.Config} | Speaker: {args.Speaker} | Whisper: {args.WhisperVersion}")
    task = configs[args.Config]

    result = {
        "config": task.name,
        "whisper_version": args.WhisperVersion,
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
        result.update(run_task_for_speaker(task, args.Speaker, args.WhisperVersion))
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