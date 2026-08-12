from training_pipeline import WhisperTrainingPipeline

pipeline = WhisperTrainingPipeline(
    intermediate_dataset_dir="...",
    finetune_dataset_path="...",
    speakers=["spk01", "spk03"],   # None = use every speaker found
)

pipeline.run_intermediate()                 # stage 1 only
pipeline.run_finetune()                     # stage 2 only, starts from intermediate_output_dir by default
pipeline.run_finetune(base_model_dir="...") # or start fine-tuning from any checkpoint
pipeline.run_pipeline()                     # runs both in sequence


""" Or from terminal:

python train_whisper.py                                   # full pipeline
python train_whisper.py --stage intermediate
python train_whisper.py --stage finetune --speakers spk01 spk03
python train_whisper.py --stage finetune --base-model-dir /path/to/some/checkpoint
python train_whisper.py --finetune-dataset-path /other/dataset --finetune-num-train-epochs 15

"""
