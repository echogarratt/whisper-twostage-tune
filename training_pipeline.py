import os, glob, gc, argparse
import numpy as np
import torch, soundfile as sf, librosa, pandas as pd
from dataclasses import dataclass
from typing import Any
from datetime import timedelta
from accelerate import PartialState, InitProcessGroupKwargs
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    WhisperTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback,
)
from datasets import Dataset
import evaluate

torch.set_num_threads(8)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# CONFIG
# ============================================================
DEFAULT_MODEL_ID = "openai/whisper-small"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SPEAKERS = ["F000", "M001", "F002", "M003", "M004", "M005"]

DEFAULT_MAX_AUDIO_SECONDS = 30

DEFAULT_INTERMEDIATE_DATASET_DIR = "/mnt/parscratch/users/acd22dg/dataset-transformer"
DEFAULT_INTERMEDIATE_METADATA_PATH = "/mnt/parscratch/users/acd22dg/dataset-transformer/metadata.csv"
DEFAULT_INTERMEDIATE_OUTPUT_DIR = "/mnt/parscratch/users/acd22dg/whisper-intermediate-output"
DEFAULT_INTERMEDIATE_NUM_TRAIN_EPOCHS = 4

DEFAULT_FINETUNE_DATASET_PATH = "/mnt/parscratch/users/acd22dg/dataset-original"
DEFAULT_FINETUNE_METADATA_PATH = os.path.join(DEFAULT_FINETUNE_DATASET_PATH, "metadata.csv")
DEFAULT_FINETUNE_OUTPUT_DIR = "/mnt/parscratch/users/acd22dg/whisper-model-output"
DEFAULT_FINETUNE_NUM_TRAIN_EPOCHS = 30
DEFAULT_FINETUNE_EARLY_STOPPING_PATIENCE = 3


def prepare_audio_batch(batch, feature_extractor, tokenizer, id_to_transcript, sample_rate):
    feats, labels = [], []
    for path in batch["file"]:
        utt_id = os.path.splitext(os.path.basename(path))[0]
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        features = feature_extractor(
            audio,
            sampling_rate=sample_rate
        ).input_features[0]
        feats.append(features)
        labels.append(tokenizer(id_to_transcript[utt_id]).input_ids)
    return {"input_features": feats, "labels": labels}


# ============================================================
# DATA COLLATOR
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels

        return batch


# ============================================================
# TRAINING PIPELINE CLASS
# ============================================================
class WhisperTrainingPipeline:
    def __init__(self, model_id=DEFAULT_MODEL_ID, sample_rate=DEFAULT_SAMPLE_RATE,
                 max_audio_seconds=DEFAULT_MAX_AUDIO_SECONDS,
                 intermediate_dataset_dir=DEFAULT_INTERMEDIATE_DATASET_DIR,
                 intermediate_metadata_path=DEFAULT_INTERMEDIATE_METADATA_PATH,
                 intermediate_output_dir=DEFAULT_INTERMEDIATE_OUTPUT_DIR,
                 intermediate_num_train_epochs=DEFAULT_INTERMEDIATE_NUM_TRAIN_EPOCHS,
                 finetune_dataset_path=DEFAULT_FINETUNE_DATASET_PATH,
                 finetune_metadata_path=DEFAULT_FINETUNE_METADATA_PATH,
                 finetune_output_dir=DEFAULT_FINETUNE_OUTPUT_DIR,
                 finetune_num_train_epochs=DEFAULT_FINETUNE_NUM_TRAIN_EPOCHS,
                 finetune_early_stopping_patience=DEFAULT_FINETUNE_EARLY_STOPPING_PATIENCE,
                 speakers=None):
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds

        self.intermediate_dataset_dir = intermediate_dataset_dir
        self.intermediate_metadata_path = intermediate_metadata_path
        self.intermediate_output_dir = intermediate_output_dir
        self.intermediate_num_train_epochs = intermediate_num_train_epochs

        self.finetune_dataset_path = finetune_dataset_path
        self.finetune_metadata_path = finetune_metadata_path
        self.finetune_output_dir = finetune_output_dir
        self.finetune_num_train_epochs = finetune_num_train_epochs
        self.finetune_early_stopping_patience = finetune_early_stopping_patience

        self.speakers = speakers if speakers is not None else DEFAULT_SPEAKERS

        kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=1))
        self.state = PartialState(kwargs_handlers=[kwargs])

        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_id, language="English", task="transcribe")
        self.metric = evaluate.load("wer")

        self.id_to_transcript = {}
        for path in filter(None, [self.finetune_metadata_path, self.intermediate_metadata_path]):
            if os.path.exists(path):
                df = pd.read_csv(path)
                if "utterance_id" in df.columns and "transcript" in df.columns:
                    self.id_to_transcript.update(zip(df["utterance_id"].astype(str), df["transcript"].astype(str)))

# ============================================================
# DATA PROCESSING HELPERS
# ============================================================

    def load_audio(self, path):
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio

    def filter_missing_transcripts(self, files, source_context=""):
        valid_files = []
        skipped_count = 0
        for path in files:
            utt_id = os.path.splitext(os.path.basename(path))[0]
            if utt_id in self.id_to_transcript:
                valid_files.append(path)
            else:
                skipped_count += 1
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files missing transcripts from {source_context}")
        return valid_files

    def make_dataset(self, files):
        with self.state.local_main_process_first():
            ds = Dataset.from_dict({"file": files}).map(
                prepare_audio_batch,
                fn_kwargs={
                    "feature_extractor": self.processor.feature_extractor,
                    "tokenizer": self.tokenizer,
                    "id_to_transcript": self.id_to_transcript,
                    "sample_rate": self.sample_rate,
                },
                batched=True,
                batch_size=64,
                num_proc=8,
                remove_columns=["file"]
            )
        return ds

    def compute_metrics(self, pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        pred_str = self.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * self.metric.compute(predictions=pred_str, references=label_str)}

    def _find_dir_case_insensitive(self, base_path, target_name):
        if not base_path or not os.path.exists(base_path):
            return None
        for item in os.listdir(base_path):
            if item.lower() == target_name.lower():
                full_path = os.path.join(base_path, item)
                if os.path.isdir(full_path):
                    return full_path
        return None

    def load_speaker_data(self, base_path):
        data = {}
        if not base_path or not os.path.exists(base_path):
            return data
        for item in os.listdir(base_path):
            spk_path = os.path.join(base_path, item)
            if not os.path.isdir(spk_path): continue

            train_dir = self._find_dir_case_insensitive(spk_path, "train")
            test_dir = self._find_dir_case_insensitive(spk_path, "test")
            train_files = sorted(glob.glob(os.path.join(train_dir, "*.wav"))) if train_dir else []
            test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav"))) if test_dir else []

            if not train_files and not test_files:
                all_files = sorted(list(set(
                    glob.glob(os.path.join(spk_path, "*.wav")) +
                    glob.glob(os.path.join(spk_path, "**", "*.wav"), recursive=True)
                )))
                split_idx = int(len(all_files) * 0.9)
                train_files = all_files[:split_idx]
                test_files = all_files[split_idx:]

            data[item.upper()] = {"train": train_files, "test": test_files, "original_name": item}
        return data

    @staticmethod
    def _configure_model(model, freeze_decoder=False, freeze_encoder=False):
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        model.generation_config.language = "english"
        model.generation_config.task = "transcribe"
        model.generation_config.num_beams = 1
        model.config.use_cache = False

        # Configure encoder freezing
        for p in model.model.encoder.parameters():
            p.requires_grad = not freeze_encoder

        # Configure decoder freezing (includes proj_out to avoid partial freezing)
        for p in model.model.decoder.parameters():
            p.requires_grad = not freeze_decoder
        if hasattr(model, "proj_out"):
            for p in model.proj_out.parameters():
                p.requires_grad = not freeze_decoder

        return model

# ============================================================
# INTERMEDIATE STAGE
# ============================================================

    def run_intermediate(self, target_speaker=None):
        if target_speaker is None:
            results = {}
            for spk in self.speakers:
                results[spk] = self.run_intermediate(target_speaker=spk)
            return results

        if not self.intermediate_dataset_dir:
            return self.model_id

        spk_dir = self._find_dir_case_insensitive(self.intermediate_dataset_dir, target_speaker)
        if not spk_dir:
            print(f"Directory for speaker '{target_speaker}' not found in {self.intermediate_dataset_dir}")
            return self.model_id

        train_dir = self._find_dir_case_insensitive(spk_dir, "train")
        test_dir = self._find_dir_case_insensitive(spk_dir, "test")

        if train_dir and test_dir:
            train_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
            eval_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
        else:
            all_files = sorted(list(set(
                glob.glob(os.path.join(spk_dir, "*.wav")) +
                glob.glob(os.path.join(spk_dir, "**", "*.wav"), recursive=True)
            )))
            split_idx = int(len(all_files) * 0.9)
            train_files = all_files[:split_idx]
            eval_files = all_files[split_idx:]

        train_files = self.filter_missing_transcripts(train_files, source_context=f"intermediate train datasets for {target_speaker}")
        eval_files = self.filter_missing_transcripts(eval_files, source_context=f"intermediate eval datasets for {target_speaker}")

        if not train_files:
            print(f"No wav files found for speaker '{target_speaker}' in {spk_dir}")
            return self.model_id

        output_dir = f"{self.intermediate_output_dir}/{target_speaker}"
        print(f"Intermediate stage for speaker {target_speaker}: {len(train_files)} train files, {len(eval_files)} eval files")

        intermediate_train_ds = self.make_dataset(train_files)
        intermediate_eval_ds = self.make_dataset(eval_files) if eval_files else None

        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            attn_implementation="sdpa"
        )

        # Decoder frozen: intermediate stage only adapts the encoder to the
        # (synthetic) target-domain acoustics; the decoder's pretrained
        # language model is left untouched.
        model = self._configure_model(model, freeze_decoder=True, freeze_encoder=False)
        collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor, model.config.decoder_start_token_id)

        # whisper-small + decoder frozen means substantially less memory
        # pressure than the old whisper-medium/full-fine-tune setup (no
        # Adam optimizer state for the decoder, smaller model overall), so
        # we can push per-device batch size up while keeping the same
        # effective batch size via fewer accumulation steps - same training
        # dynamics, less accumulation overhead.
        per_device_batch = 32
        grad_accum_steps = 2
        effective_batch_size = per_device_batch * grad_accum_steps  # 64, same as before
        steps_per_epoch = max(1, len(intermediate_train_ds) // effective_batch_size)

        intermediate_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum_steps,
            gradient_checkpointing=True,
            # Only the encoder is trainable now (decoder frozen). LR 1e-5
            # was tuned for full unfrozen fine-tuning of whisper-medium;
            # with a much smaller trainable-parameter subset on
            # whisper-small, bumping this up gives the encoder a real
            # chance to move within a fixed epoch budget.
            learning_rate=3e-5,
            warmup_steps=min(500, steps_per_epoch * self.intermediate_num_train_epochs // 5),
            num_train_epochs=self.intermediate_num_train_epochs,
            bf16=True,
            tf32=True,
            torch_compile=False,
            # Still eval per epoch for visibility into synthetic-domain WER,
            # but no longer used to pick a checkpoint (see below).
            eval_strategy="epoch" if intermediate_eval_ds else "no",
            # No early stopping, so no need to keep per-epoch checkpoints
            # around either - just save whatever's in memory at the end.
            save_strategy="no",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            # load_best_model_at_end intentionally OFF: it was selecting
            # checkpoints by WER on the synthetic eval split, which may not
            # correlate with real-speaker performance - that mismatch is a
            # live suspect for why intermediate tuning isn't helping. Take
            # the final-epoch model instead of second-guessing which epoch
            # was "best" on a metric we don't trust yet.
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=False,
            dataloader_drop_last=True,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=2,
            dataloader_persistent_workers=True,
            optim="adamw_torch_fused",
            ddp_find_unused_parameters=False,
            report_to=[],
            ddp_timeout=3600,
        )

        # Early stopping removed: with load_best_model_at_end off and no
        # trusted stopping metric, just run the full configured epoch
        # budget every time.
        callbacks = None

        intermediate_trainer = Seq2SeqTrainer(
            model=model, args=intermediate_args,
            train_dataset=intermediate_train_ds,
            eval_dataset=intermediate_eval_ds,
            data_collator=collator,
            compute_metrics=self.compute_metrics if intermediate_eval_ds else None,
            processing_class=self.processor.feature_extractor,
            callbacks=callbacks,
        )

        intermediate_trainer.train()
        intermediate_trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        del intermediate_trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        finetune_speaker_data = self.load_speaker_data(self.finetune_dataset_path)
        if not finetune_speaker_data:
            return self.run_inference(base_model_dir=output_dir, target_speakers=[target_speaker])

        return output_dir

# ============================================================
# FINE-TUNING STAGE
# ============================================================

    def prepare_finetune_dataset(self, target, speaker_data):
        target_key = target.upper()
        if target_key not in speaker_data:
            return None, None

        spk_info = speaker_data[target_key]
        train_files = self.filter_missing_transcripts(spk_info["train"], source_context=f"finetune train set for {target}")
        eval_files = self.filter_missing_transcripts(spk_info["test"], source_context=f"finetune test set for {target}")

        if not train_files or not eval_files:
            return None, None

        return self.make_dataset(train_files), self.make_dataset(eval_files)

    def _finetune_single_speaker(self, target, speaker_data, base_model_dir, train_ds=None, eval_ds=None):
        target_key = target.upper()
        if target_key not in speaker_data:
            return None

        if train_ds is None or eval_ds is None:
            train_ds, eval_ds = self.prepare_finetune_dataset(target, speaker_data)
            if train_ds is None:
                return None

        print(f"\n================ Fine-Tuning Speaker: {target} ================")

        spk_base_model = f"{base_model_dir}/{target}" if os.path.exists(f"{base_model_dir}/{target}") else base_model_dir

        model = WhisperForConditionalGeneration.from_pretrained(
            spk_base_model,
            attn_implementation="sdpa"
        )

        # Decoder frozen: fine-tune stage only adapts the encoder to the
        # real target speaker's acoustics; the decoder's language model
        # stays as pretrained (or as left by the intermediate stage).
        model = self._configure_model(model, freeze_decoder=True, freeze_encoder=False)

        processor_ft = WhisperProcessor.from_pretrained(spk_base_model)
        finetune_collator = DataCollatorSpeechSeq2SeqWithPadding(processor_ft, model.config.decoder_start_token_id)

        # Fine-tune sets are ~90 real training utterances/speaker - an order
        # of magnitude smaller than the intermediate stage. The old config
        # reused the intermediate stage's effective batch size of 64 with
        # dataloader_drop_last=True, which meant ~1 optimizer step per
        # epoch (steps_per_epoch = 90 // 64 = 1) and silently dropped
        # whatever didn't fill a batch of 16 every epoch. Use a batch size
        # that's actually appropriate for this dataset size instead.
        finetune_per_device_batch_size = 8
        finetune_grad_accum_steps = 1
        finetune_effective_batch_size = finetune_per_device_batch_size * finetune_grad_accum_steps
        steps_per_epoch = max(1, len(train_ds) // finetune_effective_batch_size)

        finetune_args = Seq2SeqTrainingArguments(
            output_dir=f"{self.finetune_output_dir}/{target}",
            per_device_train_batch_size=finetune_per_device_batch_size,
            gradient_accumulation_steps=finetune_grad_accum_steps,
            gradient_checkpointing=True,
            # Encoder-only training here too, and weight_decay is already
            # doing some overfitting insurance. A slightly higher LR than
            # the old full-fine-tune default gives the encoder a real
            # chance to adapt within 10 epochs on ~90 examples, though kept
            # lower than the intermediate stage's LR since this dataset is
            # ~100x smaller and overfits much faster.
            learning_rate=2e-5,
            weight_decay=0.01,  # extra regularization - tiny dataset, easy to overfit
            warmup_steps=min(500, steps_per_epoch * self.finetune_num_train_epochs // 5),
            num_train_epochs=self.finetune_num_train_epochs,
            bf16=True,
            tf32=True,
            torch_compile=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            dataloader_drop_last=False,  # don't silently discard ~10% of an already tiny set
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=2,
            dataloader_persistent_workers=True,
            optim="adamw_torch_fused",
            ddp_find_unused_parameters=False,
            report_to=[],
            ddp_timeout=3600,
        )

        trainer = Seq2SeqTrainer(
            model=model, args=finetune_args, train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=finetune_collator, compute_metrics=self.compute_metrics,
            processing_class=processor_ft.feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.finetune_early_stopping_patience)],
        )

        trainer.train()

        pred_output = trainer.predict(eval_ds)
        wer = pred_output.metrics.get("test_wer", 100.0)
        print(f"WER for {target} / fine_tuned: {wer:.2f}%")

        preds = np.where(pred_output.predictions != -100, pred_output.predictions, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(pred_output.label_ids != -100, pred_output.label_ids, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        df_transcripts = pd.DataFrame({"ground_truth": decoded_labels, "prediction": decoded_preds})
        os.makedirs(f"{self.finetune_output_dir}/{target}", exist_ok=True)
        df_transcripts.to_csv(f"{self.finetune_output_dir}/{target}/transcripts_finetuned.csv", index=False)

        trainer.save_model(f"{self.finetune_output_dir}/{target}/final_model")
        del trainer, model, pred_output, preds, labels
        gc.collect()
        torch.cuda.empty_cache()

        return wer

    def run_finetune(self, base_model_dir=None, target_speaker=None):
        base_model_dir = base_model_dir or self.intermediate_output_dir
        speaker_data = self.load_speaker_data(self.finetune_dataset_path)
        results = {}

        targets = [target_speaker] if target_speaker is not None else self.speakers
        for target in targets:
            wer = self._finetune_single_speaker(target, speaker_data, base_model_dir)
            if wer is not None:
                results[target] = wer
            gc.collect()
            torch.cuda.empty_cache()

        table = pd.DataFrame.from_dict(results, orient="index", columns=["fine_tuned"])
        if not table.empty:
            table.loc["AVERAGE"] = table.mean(numeric_only=True)
            table = table.round(2)
        print("\n======= WER BY SPEAKER =======")
        print(table.to_string())

        os.makedirs(self.finetune_output_dir, exist_ok=True)
        # Avoid race conditions: write per-speaker CSV if evaluating a single target speaker
        if len(targets) == 1:
            table.to_csv(os.path.join(self.finetune_output_dir, f"wer_{targets[0]}.csv"))
        else:
            table.to_csv(os.path.join(self.finetune_output_dir, "wer_by_speaker.csv"))

        return table

# ============================================================
# INFERENCE STAGE
# ============================================================

    def run_inference(self, base_model_dir=None, target_speakers=None):
        base_model_dir = base_model_dir or self.model_id
        speaker_data = self.load_speaker_data(self.finetune_dataset_path)
        if not speaker_data:
            speaker_data = self.load_speaker_data(self.intermediate_dataset_dir)
        results = {}

        print(f"\n--> Running inference using model: {base_model_dir}")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        eval_speakers = target_speakers if target_speakers is not None else self.speakers

        processor = WhisperProcessor.from_pretrained(base_model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to(device)
        model.eval()
        model.config.use_cache = True

        model.generation_config.language = "english"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None

        for target in eval_speakers:
            target_key = target.upper()
            if target_key not in speaker_data:
                continue

            spk_info = speaker_data[target_key]
            eval_files = self.filter_missing_transcripts(spk_info["test"], source_context=f"inference test set for {target}")
            if not eval_files: continue

            print(f"================ Evaluating Speaker: {target} ================")
            predictions, references, valid_files = [], [], []

            for path in eval_files:
                utt_id = os.path.splitext(os.path.basename(path))[0]

                audio = self.load_audio(path)
                inputs = processor(
                    audio, sampling_rate=self.sample_rate, return_tensors="pt"
                ).input_features.to(device, dtype=model.dtype)

                with torch.no_grad():
                    generated_ids = model.generate(inputs, max_new_tokens=225)
                    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                predictions.append(transcription)
                references.append(self.id_to_transcript[utt_id])
                valid_files.append(os.path.basename(path))

            if predictions:
                wer = 100 * self.metric.compute(predictions=predictions, references=references)
                results[target] = wer
                print(f"WER for {target} / zero-shot: {wer:.2f}%")

                df_transcripts = pd.DataFrame({
                    "file": valid_files,
                    "ground_truth": references,
                    "prediction": predictions
                })
                os.makedirs(f"{self.finetune_output_dir}/{target}", exist_ok=True)
                df_transcripts.to_csv(f"{self.finetune_output_dir}/{target}/transcripts_zeroshot.csv", index=False)

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

        table = pd.DataFrame.from_dict(results, orient="index", columns=["zero_shot"])
        if not table.empty:
            table.loc["AVERAGE"] = table.mean(numeric_only=True)
            table = table.round(2)

        print("\n======= WER BY SPEAKER =======")
        print(table.to_string())

        os.makedirs(self.finetune_output_dir, exist_ok=True)
        # Avoid race conditions: write per-speaker CSV if evaluating a single target speaker
        if len(eval_speakers) == 1:
            table.to_csv(os.path.join(self.finetune_output_dir, f"wer_{eval_speakers[0]}.csv"))
        else:
            table.to_csv(os.path.join(self.finetune_output_dir, "wer_by_speaker.csv"))

        return table

    def run_pipeline(self):
        model_dir = self.run_intermediate()
        finetune_data = self.load_speaker_data(self.finetune_dataset_path)
        if finetune_data:
            return self.run_finetune(base_model_dir=model_dir)
        return None


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Whisper intermediate + per-speaker fine-tuning pipeline")
    parser.add_argument("--stage", choices=["intermediate", "finetune", "pipeline"], default="pipeline")

    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--max-audio-seconds", type=int, default=DEFAULT_MAX_AUDIO_SECONDS)

    parser.add_argument("--intermediate-dataset-dir", default=DEFAULT_INTERMEDIATE_DATASET_DIR)
    parser.add_argument("--intermediate-metadata-path", default=DEFAULT_INTERMEDIATE_METADATA_PATH)
    parser.add_argument("--intermediate-output-dir", default=DEFAULT_INTERMEDIATE_OUTPUT_DIR)
    parser.add_argument("--intermediate-num-train-epochs", type=int, default=DEFAULT_INTERMEDIATE_NUM_TRAIN_EPOCHS)

    parser.add_argument("--finetune-dataset-path", default=DEFAULT_FINETUNE_DATASET_PATH)
    parser.add_argument("--finetune-metadata-path", default=DEFAULT_FINETUNE_METADATA_PATH)
    parser.add_argument("--finetune-output-dir", default=DEFAULT_FINETUNE_OUTPUT_DIR)
    parser.add_argument("--finetune-num-train-epochs", type=int, default=DEFAULT_FINETUNE_NUM_TRAIN_EPOCHS)
    parser.add_argument("--finetune-early-stopping-patience", type=int, default=DEFAULT_FINETUNE_EARLY_STOPPING_PATIENCE)

    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS, help="List of target speakers to fine-tune")
    parser.add_argument("--target-speaker", default=None, help="Single target speaker; applies to --stage intermediate or finetune. Omit to loop over all --speakers in this process (not recommended for multiple speakers).")
    parser.add_argument("--base-model-dir", default=None, help="Only used with --stage finetune; defaults to --intermediate-output-dir")

    return parser.parse_args()


# ============================================================
# EXECUTION
# ============================================================
def main():
    args = parse_args()

    pipeline = WhisperTrainingPipeline(
        model_id=args.model_id,
        sample_rate=args.sample_rate,
        max_audio_seconds=args.max_audio_seconds,
        intermediate_dataset_dir=args.intermediate_dataset_dir,
        intermediate_metadata_path=args.intermediate_metadata_path,
        intermediate_output_dir=args.intermediate_output_dir,
        intermediate_num_train_epochs=args.intermediate_num_train_epochs,
        finetune_dataset_path=args.finetune_dataset_path,
        finetune_metadata_path=args.finetune_metadata_path,
        finetune_output_dir=args.finetune_output_dir,
        finetune_num_train_epochs=args.finetune_num_train_epochs,
        finetune_early_stopping_patience=args.finetune_early_stopping_patience,
        speakers=args.speakers,
    )

    print(pipeline.intermediate_metadata_path)
    if args.stage == "intermediate":
        pipeline.run_intermediate(target_speaker=args.target_speaker)
    elif args.stage == "finetune":
        pipeline.run_finetune(base_model_dir=args.base_model_dir, target_speaker=args.target_speaker)
    else:
        pipeline.run_pipeline()


if __name__ == "__main__":
    main()