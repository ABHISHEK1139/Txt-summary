import argparse
import logging
import os
import re
import time
import json
import shutil
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler,
)


logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a text summarization model with Hugging Face Transformers.'
    )
    parser.add_argument('--model_name', type=str, default='t5-small', help='Hugging Face model identifier')
    parser.add_argument('--dataset', type=str, default='cnn_dailymail', help='Dataset name (Hugging Face)')
    parser.add_argument('--dataset_config', type=str, default='3.0.0', help='Dataset config name')
    parser.add_argument('--train_split', type=str, default='train', help='Split name for training')
    parser.add_argument('--eval_split', type=str, default='validation', help='Split name for evaluation')
    parser.add_argument('--output_dir', type=str, default='../models', help='Directory to save final model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save intermediate checkpoints')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-5)  # Fine-tuning LR
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=150)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_epochs', type=int, default=1, help='Save checkpoint every X epochs')
    parser.add_argument('--save_steps', type=int, default=500, help='Save step checkpoint every X optimizer steps')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Max training samples (None=all)')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='Max eval samples (None=all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true', help='Use fp16 mixed precision training')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


# ============================================================
# Training State Management (true step-level resume)
# ============================================================
def get_state_file(checkpoint_dir):
    return os.path.join(checkpoint_dir, 'training_state.json')


def save_state(checkpoint_dir, epoch, step, global_step, checkpoint_path):
    """Save exact training position."""
    state = {
        'epoch': epoch,
        'step': step,
        'global_step': global_step,
        'checkpoint_path': checkpoint_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    state_file = get_state_file(checkpoint_dir)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    logger.info(f"State saved: epoch={epoch}, step={step}")


def load_state(checkpoint_dir):
    """Load last training position."""
    state_file = get_state_file(checkpoint_dir)
    if not os.path.isfile(state_file):
        return None
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        ckpt = state.get('checkpoint_path', '')
        if ckpt and os.path.isfile(os.path.join(ckpt, 'config.json')):
            return state
        else:
            logger.warning(f"State file points to missing checkpoint: {ckpt}")
            return None
    except Exception as e:
        logger.warning(f"Could not read state file: {e}")
        return None


def find_latest_checkpoint(checkpoint_dir):
    """Find latest checkpoint using state file as primary truth."""
    if not os.path.isdir(checkpoint_dir):
        return None, 0, 0

    contents = os.listdir(checkpoint_dir)
    logger.info(f"Checkpoint directory: {contents}")

    # PRIMARY: state file
    state = load_state(checkpoint_dir)
    if state:
        return state['checkpoint_path'], state['epoch'], state.get('step', 0)

    # FALLBACK: epoch checkpoints only
    epoch_ckpts = []
    for d in contents:
        match = re.search(r'epoch-(\d+)', d)
        if match:
            full_path = os.path.join(checkpoint_dir, d)
            if os.path.isfile(os.path.join(full_path, 'config.json')):
                epoch_ckpts.append((int(match.group(1)), full_path))

    if epoch_ckpts:
        latest = max(epoch_ckpts, key=lambda x: x[0])
        return latest[1], latest[0], 0

    # Fresh start — clean up orphaned step checkpoints
    for d in contents:
        if d.startswith('step-'):
            shutil.rmtree(os.path.join(checkpoint_dir, d), ignore_errors=True)
            logger.info(f"Deleted orphaned: {d}")

    return None, 0, 0


def tokenize_and_format(batch, tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dict[str, List[int]]:
    # FIX 1: Add T5 task prefix
    prefixed = ["summarize: " + article for article in batch['article']]
    inputs = tokenizer(
        prefixed,
        max_length=args.max_source_length,
        truncation=True,
        padding='max_length',
    )
    targets = tokenizer(
        text_target=batch['highlights'],
        max_length=args.max_target_length,
        truncation=True,
        padding='max_length',
    )

    # FIX 2: Replace padding with -100 so loss ignores it
    labels = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in targets['input_ids']
    ]

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels,
    }


def load_and_preprocess_dataset(tokenizer: AutoTokenizer, args: argparse.Namespace):
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenized_cache')
    # v2 cache forces re-tokenization with task prefix + padding mask
    train_cache = os.path.join(cache_dir, f'train_v2_{args.max_train_samples or "all"}')
    eval_cache = os.path.join(cache_dir, f'eval_v2_{args.max_eval_samples or "all"}')

    # Try loading from cache first
    if os.path.isdir(train_cache) and os.path.isdir(eval_cache):
        from datasets import load_from_disk
        logger.info(f"Loading cached tokenized data from {cache_dir}")
        tokenized_train = load_from_disk(train_cache)
        tokenized_eval = load_from_disk(eval_cache)
        tokenized_train.set_format(type='torch')
        tokenized_eval.set_format(type='torch')
        return tokenized_train, tokenized_eval

    # No cache — tokenize from scratch
    dataset_path = os.path.abspath(args.dataset)
    if os.path.isdir(dataset_path):
        logger.info(f"Loading local dataset from {dataset_path}")
        data_files = {
            args.train_split: os.path.join(dataset_path, f"{args.train_split}.csv"),
            args.eval_split: os.path.join(dataset_path, f"{args.eval_split}.csv"),
        }
        raw_datasets = load_dataset("csv", data_files=data_files)
    else:
        logger.info(f"Loading dataset from Hub ({args.dataset})")
        raw_datasets = load_dataset(args.dataset, args.dataset_config)
        
    column_names = raw_datasets[args.train_split].column_names

    train_split = raw_datasets[args.train_split]
    eval_split = raw_datasets[args.eval_split]

    # Subset the data if requested
    if args.max_train_samples is not None:
        train_split = train_split.select(range(min(args.max_train_samples, len(train_split))))
        logger.info(f"Using {len(train_split)} training samples")
    if args.max_eval_samples is not None:
        eval_split = eval_split.select(range(min(args.max_eval_samples, len(eval_split))))
        logger.info(f"Using {len(eval_split)} eval samples")

    remove_columns = column_names

    tokenized_train = train_split.map(
        lambda batch: tokenize_and_format(batch, tokenizer, args),
        batched=True,
        remove_columns=remove_columns,
    )
    tokenized_eval = eval_split.map(
        lambda batch: tokenize_and_format(batch, tokenizer, args),
        batched=True,
        remove_columns=remove_columns,
    )

    # Save to cache for future runs
    os.makedirs(cache_dir, exist_ok=True)
    tokenized_train.save_to_disk(train_cache)
    tokenized_eval.save_to_disk(eval_cache)
    logger.info(f"Saved tokenized data to {cache_dir}")

    tokenized_train.set_format(type='torch')
    tokenized_eval.set_format(type='torch')

    return tokenized_train, tokenized_eval


def build_dataloader(dataset, batch_size: int, shuffle: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def evaluate(model, dataloader, tokenizer, device, args):
    """Evaluate with ROUGE scores and sample generation."""
    model.eval()
    losses = []
    predictions = []
    references = []
    eval_count = 0

    try:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge")
        has_rouge = True
    except Exception:
        has_rouge = False
        logger.warning("ROUGE evaluation not available. Install: pip install evaluate rouge_score")

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_device)
            losses.append(outputs.loss.item())

            if eval_count < 200:  # Generate summaries for ROUGE
                generated = model.generate(
                    batch_device['input_ids'],
                    attention_mask=batch_device['attention_mask'],
                    max_length=args.max_target_length,
                    num_beams=4,
                    length_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
                decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                # Replace -100 with pad token for decoding
                labels = batch['labels'].clone()
                labels[labels == -100] = tokenizer.pad_token_id
                decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_refs)
                eval_count += len(decoded_preds)

    model.train()
    average_loss = sum(losses) / len(losses) if losses else float('nan')

    # Compute ROUGE
    rouge_scores = None
    if has_rouge and predictions:
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        logger.info(f'ROUGE-1: {rouge_scores["rouge1"]:.4f} | ROUGE-2: {rouge_scores["rouge2"]:.4f} | ROUGE-L: {rouge_scores["rougeL"]:.4f}')

    # Print samples
    for i in range(min(2, len(predictions))):
        logger.info(f'Sample {i+1} Model: {predictions[i][:150]}')
        logger.info(f'Sample {i+1} Ref:   {references[i][:150]}')

    return average_loss, rouge_scores


def save_checkpoint(model, tokenizer, base_dir: str, name: str):
    checkpoint_dir = os.path.abspath(os.path.join(base_dir, name))
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        saved_files = os.listdir(checkpoint_dir)
        logger.info(f'✅ SAVED checkpoint to {checkpoint_dir} ({len(saved_files)} files)')
        return checkpoint_dir
    except Exception as e:
        logger.error(f'❌ FAILED to save checkpoint to {checkpoint_dir}: {e}')
        return None


def train_loop(model, tokenizer, train_loader, eval_loader, args, start_epoch, resume_step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    remaining_epochs = args.num_train_epochs - start_epoch
    total_training_steps = len(train_loader) * remaining_epochs // args.gradient_accumulation_steps
    scheduler = get_scheduler(
        'linear', optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    global_step = 0
    model.train()

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    logger.info(f'Steps per epoch: {steps_per_epoch}')
    logger.info(f'Total optimizer steps: {total_training_steps}')
    logger.info(f'Saving checkpoint every {args.save_epochs} epoch(s) + every {args.save_steps} steps')
    if resume_step > 0:
        logger.info(f'⏩ Will skip to step {resume_step} in first epoch')

    training_start = time.time()

    for epoch in range(start_epoch + 1, args.num_train_epochs + 1):
        epoch_loss = 0.0
        epoch_start = time.time()
        skipped = 0
        batch_count = 0

        for step, batch in enumerate(train_loader, 1):
            # Skip already-processed batches on resume
            if epoch == start_epoch + 1 and resume_step > 0 and step <= resume_step:
                skipped += 1
                if skipped % 200 == 0:
                    logger.info(f'  ⏩ Skipping... {skipped}/{resume_step} batches')
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            batch_count += 1

            if step % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    elapsed = time.time() - training_start
                    speed = global_step / elapsed if elapsed > 0 else 0
                    remaining = (total_training_steps - global_step) / speed if speed > 0 else 0
                    hrs, rem = divmod(int(remaining), 3600)
                    mins, secs = divmod(rem, 60)

                    logger.info(
                        f'Epoch {epoch}/{args.num_train_epochs} step {global_step}/{total_training_steps} | '
                        f'loss={loss.item():.4f} | lr={scheduler.get_last_lr()[0]:.6f} | '
                        f'ETA={hrs}h{mins:02d}m{secs:02d}s'
                    )

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_loss, rouge_scores = evaluate(model, eval_loader, tokenizer, device, args)
                    logger.info(f'Eval loss after step {global_step}: {eval_loss:.4f}')

                # Step checkpoint with state tracking
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_name = f'step-E{epoch}-S{step}'
                    ckpt_path = save_checkpoint(model, tokenizer, args.checkpoint_dir, ckpt_name)
                    if ckpt_path:
                        save_state(args.checkpoint_dir, epoch=start_epoch, step=step, global_step=global_step, checkpoint_path=ckpt_path)
                    # Keep only last 2 step checkpoints
                    step_ckpts = sorted([
                        d for d in os.listdir(args.checkpoint_dir) if d.startswith('step-')
                    ])
                    while len(step_ckpts) > 2:
                        old = os.path.join(args.checkpoint_dir, step_ckpts.pop(0))
                        shutil.rmtree(old, ignore_errors=True)

        # Reset resume_step after first resumed epoch
        resume_step = 0

        avg_loss = epoch_loss / max(batch_count, 1)
        epoch_time = time.time() - epoch_start
        e_mins, e_secs = divmod(int(epoch_time), 60)
        logger.info(f'✅ Epoch {epoch}/{args.num_train_epochs} done | avg_loss={avg_loss:.4f} | time={e_mins}m{e_secs:02d}s')

        if args.save_epochs > 0 and epoch % args.save_epochs == 0:
            ckpt_path = save_checkpoint(model, tokenizer, args.checkpoint_dir, f'epoch-{epoch}')
            if ckpt_path:
                save_state(args.checkpoint_dir, epoch=epoch, step=0, global_step=global_step, checkpoint_path=ckpt_path)
            
    logger.info(f'Saving final model...')
    save_checkpoint(model, tokenizer, args.output_dir, 'final')

    total_time = time.time() - training_start
    t_hrs, t_rem = divmod(int(total_time), 3600)
    t_mins, t_secs = divmod(t_rem, 60)
    logger.info(f'🎉 Training complete! Total time: {t_hrs}h{t_mins:02d}m{t_secs:02d}s')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    set_seed(args.seed)

    # Auto-detect resume
    resume_path, start_epoch, resume_step = find_latest_checkpoint(args.checkpoint_dir)

    if resume_path:
        logger.info(f"🔄 RESUMING from: {resume_path}")
        logger.info(f"   Completed epoch: {start_epoch}, resume step: {resume_step}")
        if start_epoch >= args.num_train_epochs:
            logger.info(f"   ✅ Already trained {args.num_train_epochs} epochs! Delete checkpoints to retrain.")
            return
        # Clean old step checkpoints
        for d in os.listdir(args.checkpoint_dir):
            if d.startswith('step-'):
                old_path = os.path.join(args.checkpoint_dir, d)
                if old_path != resume_path:
                    shutil.rmtree(old_path, ignore_errors=True)
                    logger.info(f"   🗑️ Cleaned: {d}")
        model = AutoModelForSeq2SeqLM.from_pretrained(resume_path)
    else:
        logger.info(f"🆕 Starting fresh training")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenizer = make_tokenizer(args.model_name)
    tokenized_train, tokenized_eval = load_and_preprocess_dataset(tokenizer, args)

    train_loader = build_dataloader(
        tokenized_train, batch_size=args.per_device_train_batch_size, shuffle=True
    )
    eval_loader = build_dataloader(
        tokenized_eval, batch_size=args.per_device_eval_batch_size
    )

    train_loop(model, tokenizer, train_loader, eval_loader, args, start_epoch, resume_step)


if __name__ == '__main__':
    main()
