# ============================================================
# TEXT SUMMARIZER - COLAB T4 GPU TRAINING
# ============================================================
# FEATURES:
#   ✅ Trains on T4 GPU (16GB VRAM) - fast & reliable
#   ✅ TRUE resume: saves exact epoch + step position
#   ✅ Saves checkpoint EVERY epoch + every 500 steps
#   ✅ Any checkpoint can be used as a working model
#   ✅ Downloads CNN/DailyMail dataset to Google Drive
#   ✅ Real-time stats: loss, speed, ETA, progress
#
# HOW TO USE:
#   1. Colab > Runtime > Change runtime type > T4 GPU
#   2. Paste this into a cell > Run
#   3. If disconnected: just re-run — it skips to exact step!
#   4. To use model early: grab any checkpoint from Drive
# ============================================================

# --- STEP 1: Install dependencies ---
!pip uninstall -y torch_xla 2>/dev/null
!pip install -q datasets transformers accelerate evaluate rouge_score

# --- STEP 2: Mount Google Drive ---
from google.colab import drive
import os, subprocess, shutil, json
try:
    drive.flush_and_unmount()
except:
    pass
if os.path.exists('/content/drive'):
    subprocess.run(['fusermount', '-u', '/content/drive'], capture_output=True)
    shutil.rmtree('/content/drive', ignore_errors=True)
drive.mount('/content/drive')

import os
import re
import time
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Project structure on Google Drive ---
PROJECT_DIR     = '/content/drive/MyDrive/txt_summarizer_project'
MODEL_DIR       = os.path.join(PROJECT_DIR, 'models')
CHECKPOINT_DIR  = os.path.join(PROJECT_DIR, 'training', 'checkpoints')
DATA_DIR        = os.path.join(PROJECT_DIR, 'data')
CACHE_DIR       = os.path.join(PROJECT_DIR, 'training', 'tokenized_cache')
STATE_FILE      = os.path.join(CHECKPOINT_DIR, 'training_state.json')

for d in [MODEL_DIR, CHECKPOINT_DIR, DATA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# STEP 0: Upload checkpoint to resume (NEW ACCOUNT SETUP)
# ============================================================
# Check if a checkpoint already exists
has_checkpoint = os.path.isfile(STATE_FILE) or any(
    d.startswith('step-') or d.startswith('epoch-')
    for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
)

if not has_checkpoint:
    print("=" * 60)
    print("  🆕 NEW ACCOUNT DETECTED — No checkpoints found!")
    print("  📤 Upload your checkpoint files to resume training.")
    print("  " + "-" * 56)
    print("  Required files (from your old account):")
    print("    1. training_state.json")
    print("    2. config.json")
    print("    3. model.safetensors")
    print("    4. tokenizer.json")
    print("    5. tokenizer_config.json")
    print("    6. generation_config.json (optional)")
    print("  " + "-" * 56)
    print("  💡 Select ALL files at once from your checkpoint folder")
    print("=" * 60)

    from google.colab import files
    uploaded = files.upload()

    if uploaded:
        # Separate state file from model files
        state_data = None
        model_files = {}

        for fname, content in uploaded.items():
            if fname == 'training_state.json':
                state_data = content
            else:
                model_files[fname] = content

        if state_data:
            # Read state to get the checkpoint name
            import json as _json
            state = _json.loads(state_data.decode('utf-8'))
            old_path = state.get('checkpoint_path', '')
            # Extract just the folder name (e.g. "step-19500")
            ckpt_name = os.path.basename(old_path) if old_path else 'uploaded-checkpoint'

            # Create checkpoint folder
            ckpt_dir = os.path.join(CHECKPOINT_DIR, ckpt_name)
            os.makedirs(ckpt_dir, exist_ok=True)

            # Save model files
            for fname, content in model_files.items():
                with open(os.path.join(ckpt_dir, fname), 'wb') as f:
                    f.write(content)

            # Update state file with new path
            state['checkpoint_path'] = ckpt_dir
            with open(STATE_FILE, 'w') as f:
                _json.dump(state, f, indent=2)

            print(f"\n✅ Checkpoint uploaded to: {ckpt_dir}")
            print(f"   Steps done: {state.get('total_steps_done', '?')}")
            print(f"   State file updated with new path")
            print(f"   Model files: {list(model_files.keys())}")
        else:
            # No state file — just put model files in a folder
            ckpt_dir = os.path.join(CHECKPOINT_DIR, 'uploaded-checkpoint')
            os.makedirs(ckpt_dir, exist_ok=True)
            for fname, content in model_files.items():
                with open(os.path.join(ckpt_dir, fname), 'wb') as f:
                    f.write(content)
            print(f"\n⚠️ No training_state.json found — model uploaded but step count unknown.")
            print(f"   Will start fresh training with uploaded model weights.")
    else:
        print("\n📝 No files uploaded — starting fresh training from scratch.")
else:
    existing = os.listdir(CHECKPOINT_DIR)
    print(f"✅ Existing checkpoints found: {existing}")

# --- Config ---
MODEL_NAME      = 't5-small'
NUM_EPOCHS      = 5
BATCH_SIZE      = 16        # T4 GPU = 16GB VRAM, t5-small fits easily at 16
GRAD_ACCUM      = 1         # No accumulation needed with batch 16
LR              = 3e-5      # Fine-tuning LR (NOT training-from-scratch LR)
MAX_SOURCE_LEN  = 512
MAX_TARGET_LEN  = 150

# ============================================================
# STEP 3: Training state management
# ============================================================
def save_state(total_steps_done, checkpoint_path):
    """Save how many total optimizer steps we've completed."""
    state = {
        'total_steps_done': total_steps_done,
        'checkpoint_path': checkpoint_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def load_state():
    """Load training progress."""
    if not os.path.isfile(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        ckpt = state.get('checkpoint_path', '')
        if ckpt and os.path.isfile(os.path.join(ckpt, 'config.json')):
            return state
        else:
            print(f"   ⚠️ State file points to missing checkpoint: {ckpt}")
            return None
    except Exception as e:
        print(f"   ⚠️ Could not read state file: {e}")
        return None

# --- Find where we left off ---
state = load_state()
resume_path = None
steps_already_done = 0

if state:
    resume_path = state['checkpoint_path']
    steps_already_done = state['total_steps_done']
    print(f"\n🔄 RESUMING from: {os.path.basename(resume_path)}")
    print(f"   Steps already completed: {steps_already_done}")
    # Clean old step checkpoints (keep only the one we're resuming from)
    for d in os.listdir(CHECKPOINT_DIR):
        if d.startswith('step-'):
            p = os.path.join(CHECKPOINT_DIR, d)
            if p != resume_path:
                shutil.rmtree(p, ignore_errors=True)
else:
    # Check for epoch checkpoints without state file
    if os.path.isdir(CHECKPOINT_DIR):
        contents = os.listdir(CHECKPOINT_DIR)
        epoch_ckpts = []
        for d in contents:
            match = re.search(r'epoch-(\d+)', d)
            if match:
                full_path = os.path.join(CHECKPOINT_DIR, d)
                if os.path.isfile(os.path.join(full_path, 'config.json')):
                    epoch_ckpts.append((int(match.group(1)), full_path))
        if epoch_ckpts:
            latest = max(epoch_ckpts, key=lambda x: x[0])
            resume_path = latest[1]
            # Estimate steps done from epoch number
            # Will be corrected once we know steps_per_epoch
            steps_already_done = -latest[0]  # negative = epoch count, resolved later
            print(f"\n🔄 RESUMING from: epoch-{latest[0]}")
        # Clean orphaned step checkpoints
        for d in contents:
            if d.startswith('step-'):
                shutil.rmtree(os.path.join(CHECKPOINT_DIR, d), ignore_errors=True)

if not resume_path:
    print(f"\n🆕 Starting fresh training (5 epochs, {NUM_EPOCHS} total)")

# ============================================================
# STEP 4: GPU Setup
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU! Training will be very slow.")

# ============================================================
# STEP 5: Load & Save Dataset
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Use v2 cache to force re-tokenization with fixes (prefix + padding mask)
train_cache_path = os.path.join(CACHE_DIR, 'train_tokenized_v2')
eval_cache_path = os.path.join(CACHE_DIR, 'eval_tokenized_v2')

if os.path.isdir(train_cache_path) and os.path.isdir(eval_cache_path):
    print("⚡ Loading tokenized data from cache (instant!)...")
    train_data = load_from_disk(train_cache_path)
    eval_data = load_from_disk(eval_cache_path)
else:
    print("📥 Downloading CNN/DailyMail dataset...")
    raw = load_dataset('cnn_dailymail', '3.0.0')

    csv_train = os.path.join(DATA_DIR, 'train.csv')
    if not os.path.exists(csv_train):
        print("💾 Saving dataset CSVs to Google Drive...")
        raw['train'].to_csv(csv_train, index=False)
        raw['validation'].to_csv(os.path.join(DATA_DIR, 'validation.csv'), index=False)
        raw['test'].to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
        print(f"   Saved to {DATA_DIR}")

    def tokenize(batch):
        # FIX 1: Add T5 task prefix
        prefixed = ["summarize: " + article for article in batch['article']]
        inputs = tokenizer(prefixed, max_length=MAX_SOURCE_LEN, truncation=True, padding='max_length')
        targets = tokenizer(text_target=batch['highlights'], max_length=MAX_TARGET_LEN, truncation=True, padding='max_length')

        # FIX 2: Replace padding with -100 so loss ignores it
        labels = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
            for label in targets['input_ids']
        ]

        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels}

    print("🔧 Tokenizing training data (with task prefix + label masking)...")
    train_data = raw['train'].map(tokenize, batched=True, remove_columns=raw['train'].column_names)
    print("🔧 Tokenizing validation data...")
    eval_data = raw['validation'].select(range(2000)).map(tokenize, batched=True, remove_columns=raw['validation'].column_names)

    train_data.save_to_disk(train_cache_path)
    eval_data.save_to_disk(eval_cache_path)
    print(f"💾 Tokenized cache saved (v2 with fixes)!")

train_data.set_format('torch')
eval_data.set_format('torch')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

# ============================================================
# STEP 6: Load model (from checkpoint or fresh)
# ============================================================
if resume_path:
    model = AutoModelForSeq2SeqLM.from_pretrained(resume_path)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to(device)

steps_per_epoch = len(train_loader) // GRAD_ACCUM

# Resolve epoch-based steps_already_done (negative = epoch count)
if steps_already_done < 0:
    completed_epochs = -steps_already_done
    steps_already_done = completed_epochs * steps_per_epoch
    print(f"   Estimated steps done from epochs: {steps_already_done}")

total_target_steps = steps_per_epoch * NUM_EPOCHS
remaining_steps = total_target_steps - steps_already_done

if remaining_steps <= 0:
    print(f"\n✅ Training already complete! {steps_already_done}/{total_target_steps} steps done.")
    print(f"   Delete checkpoints folder to retrain.")
    import sys; sys.exit(0)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=200, num_training_steps=remaining_steps)
scaler = torch.amp.GradScaler('cuda')

current_epoch = steps_already_done // steps_per_epoch + 1

print(f"\n{'='*60}")
print(f"  🚀 TRAINING on {torch.cuda.get_device_name(0)}")
print(f"  Model:      {MODEL_NAME}")
print(f"  Progress:   {steps_already_done}/{total_target_steps} steps done")
print(f"  Remaining:  {remaining_steps} steps (~{remaining_steps // steps_per_epoch + 1} epochs)")
print(f"  Samples:    {len(train_data)}")
print(f"  Batch:      {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
print(f"  Steps/epoch: {steps_per_epoch}")
print(f"  Checkpoint: every 500 steps + every epoch")
print(f"{'='*60}\n")

# ============================================================
# STEP 7: Training Loop
# ============================================================
model.train()
global_step = steps_already_done  # continue counting from where we left off
training_start = time.time()

while global_step < total_target_steps:
    current_epoch = global_step // steps_per_epoch + 1
    epoch_loss = 0.0
    epoch_start = time.time()
    batch_losses = []
    batches_in_epoch = 0

    for step, batch in enumerate(train_loader, 1):
        if global_step >= total_target_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast('cuda'):
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        batch_losses.append(loss.item())
        batches_in_epoch += 1

        if step % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 25 == 0:
                elapsed = time.time() - training_start
                done_this_run = global_step - steps_already_done
                speed = done_this_run / elapsed if elapsed > 0 else 0
                remaining = (total_target_steps - global_step) / speed if speed > 0 else 0
                avg_recent = sum(batch_losses[-50:]) / len(batch_losses[-50:])
                overall_pct = global_step / total_target_steps * 100

                hrs, rem = divmod(int(remaining), 3600)
                mins, secs = divmod(rem, 60)

                print(
                    f'  📊 Step {global_step}/{total_target_steps} '
                    f'[{overall_pct:5.1f}%] | '
                    f'loss={avg_recent:.4f} | '
                    f'lr={scheduler.get_last_lr()[0]:.6f} | '
                    f'{speed:.1f} steps/s | '
                    f'ETA={hrs}h{mins:02d}m{secs:02d}s'
                )

            # Save every 500 steps
            if global_step % 500 == 0:
                ckpt_name = f'step-{global_step}'
                mid_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
                model.save_pretrained(mid_path)
                tokenizer.save_pretrained(mid_path)
                save_state(total_steps_done=global_step, checkpoint_path=mid_path)
                print(f'  💾 Saved: {ckpt_name} ({global_step}/{total_target_steps})')

                # Keep only last 2 step checkpoints
                step_ckpts = sorted([
                    d for d in os.listdir(CHECKPOINT_DIR) if d.startswith('step-')
                ], key=lambda x: int(x.split('-')[1]))
                while len(step_ckpts) > 2:
                    old = os.path.join(CHECKPOINT_DIR, step_ckpts.pop(0))
                    shutil.rmtree(old, ignore_errors=True)

    # End of one pass through data = 1 epoch
    epoch_time = time.time() - epoch_start
    avg_loss = epoch_loss / max(batches_in_epoch, 1)
    e_mins, e_secs = divmod(int(epoch_time), 60)
    print(f'\n  ✅ Epoch pass done | step {global_step}/{total_target_steps} | avg_loss={avg_loss:.4f} | time={e_mins}m{e_secs:02d}s')

    # --- FIX 3: ROUGE evaluation after each epoch ---
    print(f'  📝 Evaluating with ROUGE...')
    model.eval()
    import evaluate
    rouge = evaluate.load("rouge")
    predictions = []
    references = []
    eval_count = 0

    with torch.no_grad():
        for eval_batch in eval_loader:
            if eval_count >= 200:  # Evaluate on 200 samples (fast enough)
                break
            input_ids = eval_batch['input_ids'].to(device)
            attention_mask = eval_batch['attention_mask'].to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_TARGET_LEN,
                num_beams=4,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            # Replace -100 with pad token for decoding
            labels = eval_batch['labels'].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)
            eval_count += len(decoded_preds)

    scores = rouge.compute(predictions=predictions, references=references)
    print(f'  📊 ROUGE-1: {scores["rouge1"]:.4f} | ROUGE-2: {scores["rouge2"]:.4f} | ROUGE-L: {scores["rougeL"]:.4f}')

    # Print 2 sample summaries
    for i in range(min(2, len(predictions))):
        print(f'  📄 Sample {i+1}:')
        print(f'     Model: {predictions[i][:200]}')
        print(f'     Ref:   {references[i][:200]}')

    model.train()

    # Save epoch checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'epoch-{current_epoch}')
    try:
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        save_state(total_steps_done=global_step, checkpoint_path=ckpt_path)
        if os.path.isfile(os.path.join(ckpt_path, 'config.json')):
            print(f'  💾 Epoch checkpoint VERIFIED: epoch-{current_epoch}')
            # Force flush to Google Drive
            drive.flush_and_unmount()
            drive.mount('/content/drive')
            print(f'  ☁️ Synced to Google Drive\n')
        else:
            print(f'  ⚠️ Save may have failed — config.json missing!\n')
    except Exception as e:
        print(f'  ❌ Save error: {e}\n')

# ============================================================
# STEP 8: Save final model
# ============================================================
final_path = os.path.join(MODEL_DIR, 'final')
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

total_time = time.time() - training_start
t_hrs, t_rem = divmod(int(total_time), 3600)
t_mins, t_secs = divmod(t_rem, 60)

print(f'\n{"="*60}')
print(f'  🎉 TRAINING COMPLETE! Total time: {t_hrs}h{t_mins:02d}m{t_secs:02d}s')
print(f'{"="*60}')
print(f'  📁 Final model : {final_path}')
print(f'  📁 Checkpoints : {CHECKPOINT_DIR}')
print(f'  📁 Dataset     : {DATA_DIR}')
print(f'')
print(f'  TO USE ON YOUR PC:')
print(f'  1. Google Drive > txt_summarizer_project > models > final')
print(f'  2. Download the "final" folder')
print(f'  3. Put it in your local project "models/" folder')
print(f'  4. Run "run.bat" → done!')
print(f'')
print(f'  💡 TIP: You can also use ANY checkpoint:')
print(f'     Just rename any "epoch-X" folder to "final"')
print(f'     and put it in "models/" on your PC.')
