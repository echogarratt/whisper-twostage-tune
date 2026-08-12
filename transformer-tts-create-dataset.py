import argparse
import csv
import torch
from pathlib import Path
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def main():
    # 1. Setup argparse so Slurm can feed specific chunks to this script
    parser = argparse.ArgumentParser(description="Run parallel TTS inference")
    parser.add_argument("--input_txt", required=True, type=str, help="Path to the chunked text file")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory to save audio files")
    parser.add_argument("--chunk_id", required=True, type=str, help="Unique ID for this chunk (e.g., 01, 02)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.chunk_id}] Running inference on: {device.upper()}")

    # 2. Initialize TTS with float16 for A100 optimization
    tts = Text2Speech.from_pretrained(
        model_file="/mnt/parscratch/users/acd22dg/espnet/egs2/eltts/tts1/exp/tts_finetune_ljspeech_transformer/train.loss.ave.pth",
        train_config="exp/tts_finetune_ljspeech_transformer/config.yaml",
        vocoder_tag="parallel_wavegan/ljspeech_hifigan.v1",
        device=device,
        dtype="float16",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Read only the specific chunk of text assigned to this job
    with open(args.input_txt, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    rows = []

    def save_audio(path, audio_data, sample_rate):
        sf.write(path, audio_data, sample_rate)

    # 4. Use ThreadPoolExecutor for asynchronous file writes
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, text in enumerate(tqdm(lines, desc=f"Chunk {args.chunk_id}")):
            # Incorporate chunk_id into the filename to guarantee uniqueness
            utt_id = f"utt_{args.chunk_id}_{i:04d}"
            
            with torch.inference_mode():
                output = tts(text)
            
            wav = output["wav"].cpu().numpy()
            wav_path = out_dir / f"{utt_id}.wav"
            
            executor.submit(save_audio, wav_path, wav, tts.fs)
            rows.append({"utt_id": utt_id, "text": text, "wav_path": str(wav_path)})

    # 5. Write a chunk-specific metadata CSV to avoid race conditions
    csv_path = out_dir / f"metadata_{args.chunk_id}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "text", "wav_path"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"[{args.chunk_id}] Finished! Metadata saved to {csv_path}")

if __name__ == "__main__":
    main()