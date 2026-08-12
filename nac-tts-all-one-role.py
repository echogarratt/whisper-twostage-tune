import csv
import os
from datasets import Audio, Dataset, DatasetDict
from transformers import (
    CsmForConditionalGeneration,
    CsmProcessor,
    Trainer,
    TrainingArguments,
)

processor = CsmProcessor.from_pretrained("sesame/csm-1b")
model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b")
model.train()
model.codec_model.eval()

model.config.use_cache = False

DATASET_DIR = "/mnt/parscratch/users/acd22dg/dataset-original"
SPLITS = ["test", "train"]
METADATA_PATH = "/mnt/parscratch/users/acd22dg/dataset-original/metadata.csv"
SPEAKERS = ["F000", "M001", "F002", "M003"]

TRAINING_WORKDIR = "/mnt/parscratch/users/acd22dg/test-trainer"
FINAL_MODEL_DIR = "/mnt/parscratch/users/acd22dg/all-one-role"
OUTPUT_WAV_PATH = "/mnt/parscratch/users/acd22dg/output_example.wav"

file_lookup = {}
with open(METADATA_PATH, "r", newline="", encoding="utf-8") as data:
    r = csv.reader(data, delimiter=",")
    next(r)
    for row in r:
        file_lookup[row[0]] = row[1]

rows_by_split = {split: [] for split in SPLITS}

for speaker in SPEAKERS:
    for split in SPLITS:
        folder = os.path.join(DATASET_DIR, speaker, split)
        if not os.path.isdir(folder):
            print(f"Warning: missing folder {folder}, skipping")
            continue
        for f in os.listdir(folder):
            f_name, extension = os.path.splitext(f)
            if f_name not in file_lookup:
                print(f"Warning: {f_name} not found in metadata, skipping")
                continue
            rows_by_split[split].append(
                {
                    "audio": os.path.join(folder, f_name + ".wav"),
                    "normalized_text": file_lookup[f_name],
                    "speaker": int(SPEAKERS.index(speaker)),
                }
            )

for split, rows in rows_by_split.items():
    print(f"Split '{split}': {len(rows)} examples")
    if len(rows) == 0:
        raise RuntimeError(
            f"No examples found for split '{split}' — check DATASET_DIR/SPEAKERS/metadata paths"
        )

dataset = DatasetDict(
    {split: Dataset.from_list(rows) for split, rows in rows_by_split.items()}
)

dataset = dataset.cast_column(
    "audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate)
)


def data_collator(samples):
    conversations = []

    for sample in samples:
        role = str(int(sample["speaker"]))
        conversation = [
            {
                "role": role,
                "content": [
                    {"type": "text", "text": sample["normalized_text"]},
                    {"type": "audio", "audio": sample["audio"]["array"]},
                ],
            }
        ]
        conversations.append(conversation)

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        output_labels=True,
    )
    return inputs


training_args = TrainingArguments(
    output_dir=TRAINING_WORKDIR,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

trainer.train()

print(f"\nSaving final best model to {FINAL_MODEL_DIR} ...")
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print("Final model saved.")

print("\nTraining complete. Generating example WAV file...")
model.eval()

test_conversation = [
    {
        "role": str(1),
        "content": [
            {
                "type": "text",
                "text": "Hello, this is an automated test audio generation from the fine-tuned CSM model.",
            }
        ],
    }
]

inference_inputs = processor.apply_chat_template(
    test_conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

generated_audio = model.generate(**inference_inputs, output_audio=True)

processor.save_audio(generated_audio, OUTPUT_WAV_PATH)
print(f"Success! Generated audio saved to: {OUTPUT_WAV_PATH}")