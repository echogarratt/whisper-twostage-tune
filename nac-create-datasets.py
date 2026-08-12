import csv
import os

import torch
from tqdm import tqdm
from transformers import CsmForConditionalGeneration, CsmProcessor


# ==================== CONFIG ======================
ALL_ONE_MODEL = "/mnt/parscratch/users/acd22dg/all-one-role"
SPK_PERS_MODEL = "/mnt/parscratch/users/acd22dg/speaker-personalised"
OUTPUT_PATH = "/mnt/parscratch/users/acd22dg/datasets-final"
SPEAKERS = ["F000", "M001", "F002", "M003", "M004", "M005"]
#TODO: all one should generate roles for speakers SPEAKERS = ["F000", "M001", "F002", "M003"]
TEXT_FILE = "/mnt/parscratch/users/acd22dg/lines.txt"
BATCH_SIZE = 128
MAX_NEW_TOKENS = 1024
# ==================================================


def chunked(seq, size):
    for start in range(0, len(seq), size):
        yield list(enumerate(seq[start:start + size], start=start + 1))


def generate_batch(model, processor, texts, role, device, output_dir, start_index):
    conversations = [
        [{"role": str(role), "content": [{"type": "text", "text": text}]}]
        for text in texts
    ]

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        generated_audio = model.generate(
            **inputs,
            output_audio=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    rows = []
    for offset, (text, audio) in enumerate(zip(texts, generated_audio)):
        idx = start_index + offset
        filename = f"line_{idx:04d}.wav"
        wav_path = os.path.join(output_dir, filename)
        processor.save_audio(audio, wav_path)
        rows.append((filename, text))

    return rows


def run_for_role(model, processor, texts, role, device, output_dir, desc):
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")

    with open(metadata_path, "w", newline="", encoding="utf-8") as meta_file:
        writer = csv.writer(meta_file)

        batches = list(chunked(texts, BATCH_SIZE))
        for batch in tqdm(batches, desc=desc):
            indices = [i for i, _ in batch]
            batch_texts = [t for _, t in batch]
            start_index = indices[0]

            rows = generate_batch(
                model, processor, batch_texts, role, device, output_dir, start_index
            )
            writer.writerows(rows)
            meta_file.flush()


def load_model(model_path, device):
    print(f"Loading model from {model_path} on {device} ...")
    processor = CsmProcessor.from_pretrained(model_path, local_files_only=True)
    model = CsmForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()
    return processor, model


def main():
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"No non-empty lines found in {TEXT_FILE}. Nothing to do.")
        raise SystemExit(0)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = load_model(ALL_ONE_MODEL, device)
    processor.tokenizer.padding_side = "left"

    run_for_role(
        model, processor, lines,
        role=0,
        device=device,
        output_dir=os.path.join(OUTPUT_PATH, "all_one"),
        desc="Generating audio (all_one)",
    )

    del model
    torch.cuda.empty_cache()

    processor, model = load_model(SPK_PERS_MODEL, device)
    processor.tokenizer.padding_side = "left"

    for speaker_idx, speaker in enumerate(tqdm(SPEAKERS, desc="Speakers")):
        run_for_role(
            model, processor, lines,
            role=speaker_idx,
            device=device,
            output_dir=os.path.join(OUTPUT_PATH, speaker),
            desc=f"Generating audio ({speaker})",
        )

    print("All datasets created")


if __name__ == "__main__":
    main()