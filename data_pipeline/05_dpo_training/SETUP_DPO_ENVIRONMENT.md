# DPO Training Environment Setup Guide

**Complete reproducible setup for DPO training on Lambda Labs A100**

This guide captures all the troubleshooting and solutions discovered during the initial setup, ensuring you can reproduce the exact working environment without encountering the same issues.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Setup Script](#quick-setup-script)
3. [Manual Setup Steps](#manual-setup-steps)
4. [Critical Configuration Notes](#critical-configuration-notes)
5. [Troubleshooting Reference](#troubleshooting-reference)
6. [Model Download Instructions](#model-download-instructions)

---

## System Requirements

### Hardware
- **GPU**: NVIDIA A100 40GB (or 80GB)
- **VRAM**: ~35GB used during training
- **Disk Space**: ~50GB (models + datasets + cache)

### Software
- **OS**: Ubuntu 20.04 or 22.04
- **Python**: 3.10.x (tested on 3.10.12)
- **CUDA**: 12.1+ (for BF16 support)
- **Driver**: 525+ (compatible with CUDA 12.x)

---

## Quick Setup Script

**Use this for fastest setup** (creates virtual environment with all correct versions):

```bash
cd ~/RecipeGen-LLM/data_pipeline/05_dpo_training

# Run automated setup
./setup_dpo_env.sh

# Activate environment
source dpo_venv/bin/activate

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch: 2.5.1+cu121
CUDA Available: True
```

---

## Manual Setup Steps

### Step 1: Create Virtual Environment

```bash
cd ~/RecipeGen-LLM/data_pipeline/05_dpo_training

# Create Python 3.10 virtual environment
python3 -m venv dpo_venv

# Activate
source dpo_venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Why virtual environment?**
- Avoids system package conflicts (especially Keras 3 / TensorFlow)
- Isolated dependency versions
- Easy to recreate

### Step 2: Install Core Dependencies

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 3: Install Transformers Stack

**CRITICAL: Install in this exact order with these exact versions**

```bash
# HuggingFace ecosystem (compatible versions)
pip install transformers==4.57.1
pip install accelerate==1.12.0
pip install peft==0.18.0
pip install trl==0.25.1

# Quantization support
pip install bitsandbytes==0.45.0

# Data loading
pip install datasets==3.2.0

# Utilities
pip install pyyaml==6.0.2
```

**Version compatibility critical points:**
- `trl 0.25.1` requires `DPOConfig` (not `TrainingArguments`)
- `trl 0.25.1` requires `processing_class` parameter (not `tokenizer`)
- `accelerate 1.12.0` is compatible with latest `transformers` and `peft`
- `peft 0.18.0` works with 4-bit quantized models

### Step 4: HuggingFace Login

```bash
# Login with your HF token (for Llama 3.2 access)
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN="your_token_here"
```

**Get token:** https://huggingface.co/settings/tokens (needs "Read" access)

**Llama 3.2 access:** Request access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

### Step 5: Verify Installation

```bash
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
print('✅ All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

---

## Critical Configuration Notes

### 1. Model Loading (4-bit Quantization)

**Correct configuration:**

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0}  # CRITICAL: Direct GPU placement
)
```

**Why `device_map={"": 0}`?**
- Avoids calling `.to(device)` which fails on quantized models
- Directly places all layers on GPU 0
- `device_map="auto"` causes `ValueError: .to is not supported for 4-bit models`

### 2. LoRA Adapter Loading

**Correct configuration:**

```python
from peft import PeftModel

# Load adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# Enable gradient computation (CRITICAL!)
model.print_trainable_parameters()
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True
```

**Why explicit `requires_grad`?**
- Prevents "tensors does not require grad" error during training
- Ensures LoRA parameters are trainable
- Default loading may disable gradients

### 3. DPO Training Configuration

**Correct configuration:**

```python
from trl import DPOConfig  # NOT TrainingArguments!

training_args = DPOConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    learning_rate=5e-5,
    warmup_ratio=0.1,

    bf16=True,  # CRITICAL: Use BF16 on A100, not FP16
    gradient_checkpointing=False,  # CRITICAL: Disable for quantized models

    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
)
```

**Why BF16 instead of FP16?**
- A100 has native BF16 support (faster, more stable)
- FP16 + 4-bit quantization causes gradient scaler errors
- `AssertionError: No inf checks were recorded` resolved with BF16

**Why `gradient_checkpointing=False`?**
- Gradient checkpointing conflicts with 4-bit quantization
- Causes `RuntimeError: element 0 of tensors does not require grad`
- A100 40GB has enough VRAM without checkpointing

### 4. DPO Trainer Initialization

**Correct configuration:**

```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,  # CRITICAL: Use processing_class, not tokenizer
)

# DO NOT include these parameters (removed in trl 0.25.1):
# - beta
# - max_length
# - max_prompt_length
```

**Why `processing_class`?**
- `trl 0.25.1` renamed `tokenizer` parameter to `processing_class`
- Using `tokenizer` causes `TypeError: unexpected keyword argument 'tokenizer'`

---

## Troubleshooting Reference

### Error 1: Keras 3 Import Conflict

**Error:**
```
RuntimeError: Failed to import transformers.modeling_tf_utils because of the following error:
Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers.
```

**Solution:**
- Create virtual environment (isolate from system packages)
- Do NOT install TensorFlow in the virtual environment

### Error 2: `.to()` Not Supported for Quantized Models

**Error:**
```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
```

**Solution:**
- Use `device_map={"": 0}` instead of `device_map="auto"`
- Never call `model.to(device)` on quantized models

### Error 3: Gradient Scaler Assertion Error

**Error:**
```
AssertionError: No inf checks were recorded prior to update.
```

**Solution:**
- Change `fp16=True` to `bf16=True` in DPOConfig
- Add `gradient_checkpointing=False`

### Error 4: Tensors Do Not Require Grad

**Error:**
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**Solution:**
- Explicitly enable gradients after loading LoRA adapter:
```python
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True
```

### Error 5: DPOTrainer Unexpected Keyword Argument

**Error:**
```
TypeError: DPOTrainer.__init__() got an unexpected keyword argument 'tokenizer'
TypeError: DPOTrainer.__init__() got an unexpected keyword argument 'beta'
```

**Solution:**
- Use `DPOConfig` instead of `TrainingArguments`
- Use `processing_class=tokenizer` instead of `tokenizer=tokenizer`
- Remove `beta`, `max_length`, `max_prompt_length` parameters

### Error 6: Accelerate Version Conflicts

**Error:**
```
ImportError: cannot import name 'is_mlu_available' from 'accelerate.utils'
```

**Solution:**
- Upgrade all packages to latest compatible versions:
```bash
pip install --upgrade accelerate transformers peft trl
```

---

## Model Download Instructions

### After Training Completes

**1. Package all trained models:**

```bash
cd ~/RecipeGen-LLM

# Create tarball (~210MB for 6 personas)
tar -czf dpo_personas_trained.tar.gz models/dpo_personas/

# Verify size
ls -lh dpo_personas_trained.tar.gz
```

**2. Download to local machine:**

```bash
# On your local machine
scp ubuntu@<LAMBDA_IP>:~/RecipeGen-LLM/dpo_personas_trained.tar.gz ~/Desktop/

# Extract to project
cd /Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM
tar -xzf ~/Desktop/dpo_personas_trained.tar.gz
```

**3. Verify downloaded models:**

```bash
ls -lh models/dpo_personas/

# Should see 6 directories:
# persona_a_korean_spicy_v1.0/
# persona_b_indian_veg_v1.0/
# persona_c_italian_gf_v1.0/
# persona_d_japanese_lowsodium_v1.0/
# persona_e_mexican_vegan_v1.0/
# persona_f_chinese_keto_v1.0/

# Check adapter sizes (~35MB each)
ls -lh models/dpo_personas/*/adapter_model.safetensors
```

**4. TERMINATE Lambda instance immediately!**

Go to https://cloud.lambdalabs.com/instances and click "Terminate"

---

## Training Data Location

**On Lambda Labs:**
```
~/RecipeGen-LLM/data/dpo_training_data/
├── persona_a_korean_spicy_dpo_train.jsonl (84 pairs)
├── persona_b_indian_veg_dpo_train.jsonl (91 pairs)
├── persona_c_italian_gf_dpo_train.jsonl (88 pairs)
├── persona_d_japanese_lowsodium_dpo_train.jsonl (87 pairs)
├── persona_e_mexican_vegan_dpo_train.jsonl (88 pairs)
└── persona_f_chinese_keto_dpo_train.jsonl (84 pairs)
```

**Base LoRA adapter:**
```
~/RecipeGen-LLM/models/llama3b_lambda_lora/
```

---

## Training Command Reference

**Single persona:**
```bash
python train_dpo.py \
  --persona persona_a_korean_spicy \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --adapter ../../models/llama3b_lambda_lora \
  --data_dir ../../data/dpo_training_data \
  --output_dir ../../models/dpo_personas
```

**All personas:**
```bash
./train_all_personas.sh
```

**Expected time per persona:** 30-45 minutes
**Expected cost per persona:** $0.55-$0.83 @ $1.10/hour

---

## Re-training Checklist

Before starting DPO training on a new Lambda Labs instance:

- [ ] Launch A100 40GB instance
- [ ] SSH into instance
- [ ] Upload `dpo_training_package.tar.gz`
- [ ] Extract: `tar -xzf dpo_training_package.tar.gz`
- [ ] Run setup script: `./setup_dpo_env.sh`
- [ ] Activate environment: `source dpo_venv/bin/activate`
- [ ] Verify GPU: `nvidia-smi`
- [ ] Start tmux: `tmux new -s dpo_training`
- [ ] Run training: `./train_all_personas.sh`
- [ ] Detach from tmux: `Ctrl+B`, then `D`
- [ ] Wait for completion (~3 hours)
- [ ] Package models: `tar -czf dpo_personas_trained.tar.gz models/dpo_personas/`
- [ ] Download models via SCP
- [ ] **TERMINATE INSTANCE IMMEDIATELY**

---

## Package Versions (Tested Working Configuration)

**See `requirements_dpo_lambda.txt` for complete list**

**Core packages:**
```
torch==2.5.1+cu121
transformers==4.57.1
accelerate==1.12.0
peft==0.18.0
trl==0.25.1
bitsandbytes==0.45.0
datasets==3.2.0
```

**Python:** 3.10.12
**CUDA:** 12.1
**GPU Driver:** 525+

---

## Cost Tracking

| Task | Duration | Cost (A100 @ $1.10/hr) |
|------|----------|------------------------|
| Setup & upload | 15 min | $0.28 |
| Test 1 persona | 30-45 min | $0.55-$0.83 |
| Train 5 more personas | 2.5-3.5 hrs | $2.75-$3.85 |
| Download models | 15 min | $0.28 |
| **TOTAL** | **~4 hours** | **$3.86-$5.24** |

Add buffer: **$4.50-$6.00** total

---

## Support

**Lambda Labs:** support@lambdalabs.com

**HuggingFace TRL Issues:** https://github.com/huggingface/trl/issues

**Project Issues:** Check training logs in tmux session

---

**Last Updated:** 2025-01-21
**Tested On:** Lambda Labs A100 40GB, Ubuntu 22.04, Python 3.10.12, CUDA 12.1
