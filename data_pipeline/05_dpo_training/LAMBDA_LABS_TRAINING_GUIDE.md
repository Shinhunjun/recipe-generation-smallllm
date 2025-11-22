# Lambda Labs DPO Training Guide

## ✅ Phase 1 Complete: Data Ready!

**Status**: 522 DPO pairs formatted and packaged ✅

**Package Location**: `/Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM/dpo_training_package.tar.gz` (35MB)

**Contains**:
- Training scripts (`train_dpo.py`, `train_all_personas.sh`)
- Formatted training data (6 persona JSONL files)
- Base LoRA adapter (`llama3b_lambda_lora/`)
- Persona configuration (`personas.yaml`)

---

## 🚀 Phase 2: Lambda Labs Setup (10-15 minutes)

### Step 1: Launch A100 Instance

1. Go to https://cloud.lambdalabs.com/instances
2. Click **"Launch Instance"**
3. Select **"1x A100 (40 GB)"** - $1.10/hour
4. Choose any available region (try multiple if one is full)
5. SSH Key: Use your existing key or create new one
6. Click **"Launch"**
7. Wait ~2 minutes for instance to boot
8. **Copy the IP address** (e.g., `150.136.123.45`)

### Step 2: Connect via SSH

```bash
# Replace <LAMBDA_IP> with your instance IP
ssh ubuntu@<LAMBDA_IP>
```

**First time?** Accept the host key fingerprint (type `yes`)

### Step 3: Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install required packages (~5 minutes)
pip install torch transformers peft trl datasets accelerate bitsandbytes

# Verify GPU
nvidia-smi
```

**Expected output**: Should show `A100-SXM4-40GB`

### Step 4: Upload Training Package

**On your LOCAL machine (new terminal):**

```bash
# Replace <LAMBDA_IP> with your instance IP
scp /Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM/dpo_training_package.tar.gz ubuntu@<LAMBDA_IP>:~/
```

Upload takes ~2-3 minutes (35MB package)

**Back on Lambda instance:**

```bash
# Extract package
tar -xzf dpo_training_package.tar.gz

# Navigate to training directory
cd RecipeGen-LLM/data_pipeline/05_dpo_training

# Verify files
ls -lh
ls -lh ../../data/dpo_training_data/
ls -lh ../../models/llama3b_lambda_lora/
```

**Expected**:
- `train_dpo.py` ✅
- `train_all_personas.sh` ✅
- 6 JSONL files in `dpo_training_data/` ✅
- `llama3b_lambda_lora/` adapter ✅

---

## 🧪 Phase 3: TEST Training (30-45 minutes, $0.55-$0.83)

**IMPORTANT**: Test with 1 persona first to validate the pipeline!

### Step 1: Start tmux Session

```bash
# tmux prevents training interruption if SSH disconnects
tmux new -s dpo_training
```

**tmux Quick Reference**:
- Detach: `Ctrl+B`, then press `D`
- Reattach: `tmux attach -t dpo_training`
- Kill session: `Ctrl+B`, then type `:kill-session`

### Step 2: Train Korean Spicy Persona (Test)

```bash
python train_dpo.py \
  --persona persona_a_korean_spicy \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --adapter ../../models/llama3b_lambda_lora \
  --data_dir ../../data/dpo_training_data \
  --output_dir ../../models/dpo_personas
```

### Step 3: Monitor Training

**Watch for these signs of success**:
```
✅ Loading base model... (takes ~3-5 min)
✅ Loading LoRA adapter...
✅ Preparing dataset... (84 train, 21 eval samples)
✅ Training: [Epoch 1/3] - eval_loss decreasing
✅ Training: [Epoch 2/3] - eval_loss continues to decrease
✅ Training: [Epoch 3/3] - final eval_loss ~0.5-0.7
✅ Saving model to ../../models/dpo_personas/persona_a_korean_spicy_v1.0/
✅ Training complete!
```

**Monitor GPU usage** (new SSH session):
```bash
watch -n 1 nvidia-smi
```

Expected: ~30-35GB VRAM usage, ~90%+ GPU utilization

### Step 4: Verify Test Success

```bash
# Check model was saved
ls -lh ../../models/dpo_personas/persona_a_korean_spicy_v1.0/

# Should see:
# adapter_config.json
# adapter_model.safetensors (~35MB)
# tokenizer files
```

**If successful** ✅ → Proceed to Phase 4

**If errors** ❌ → See Troubleshooting section below

---

## 🎯 Phase 4: Train All 6 Personas (2.5-3.5 hours, $2.75-$4.17)

**Only proceed if Phase 3 was successful!**

### Option A: Automated Script (Recommended)

```bash
# Inside tmux session
./train_all_personas.sh
```

This will train all 6 personas sequentially:
1. persona_a_korean_spicy (already done in test)
2. persona_b_indian_veg
3. persona_c_italian_gf
4. persona_d_japanese_lowsodium
5. persona_e_mexican_vegan
6. persona_f_chinese_keto

**Total time**: ~2.5-3.5 hours

### Option B: Manual One-by-One

```bash
# Train each persona individually
python train_dpo.py --persona persona_b_indian_veg
python train_dpo.py --persona persona_c_italian_gf
python train_dpo.py --persona persona_d_japanese_lowsodium
python train_dpo.py --persona persona_e_mexican_vegan
python train_dpo.py --persona persona_f_chinese_keto
```

**Use Case**: If you want to monitor each persona separately

### Monitoring Progress

**Detach from tmux**: `Ctrl+B`, then `D`

**Check progress anytime**:
```bash
# Reattach to see live logs
tmux attach -t dpo_training

# Or check which models are done
ls -lh ../../models/dpo_personas/
```

**Take a break!** Training takes 2.5-3.5 hours. You can disconnect SSH safely (tmux keeps running).

---

## 📦 Phase 5: Download Models (10-15 minutes, ~$0.20)

### Step 1: Package Trained Models

```bash
# Reattach to tmux if needed
tmux attach -t dpo_training

# Navigate to project root
cd ~/RecipeGen-LLM

# Create tarball of all 6 trained models
tar -czf dpo_personas_trained.tar.gz models/dpo_personas/

# Verify size (~210MB for 6 models)
ls -lh dpo_personas_trained.tar.gz
```

### Step 2: Download to Local Machine

**On your LOCAL machine (new terminal):**

```bash
# Replace <LAMBDA_IP> with your instance IP
scp ubuntu@<LAMBDA_IP>:~/RecipeGen-LLM/dpo_personas_trained.tar.gz ~/Desktop/

# Extract to project directory
cd /Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM
tar -xzf ~/Desktop/dpo_personas_trained.tar.gz
```

Download takes ~3-5 minutes (210MB)

### Step 3: Verify Models Downloaded

```bash
# Check all 6 models are present
ls -lh models/dpo_personas/

# Should see 6 directories:
# persona_a_korean_spicy_v1.0/
# persona_b_indian_veg_v1.0/
# persona_c_italian_gf_v1.0/
# persona_d_japanese_lowsodium_v1.0/
# persona_e_mexican_vegan_v1.0/
# persona_f_chinese_keto_v1.0/

# Check adapter size (~35MB each)
ls -lh models/dpo_personas/*/adapter_model.safetensors
```

### Step 4: ⚠️ TERMINATE LAMBDA INSTANCE (CRITICAL!)

**Don't forget this step! You're charged per second!**

1. Go to https://cloud.lambdalabs.com/instances
2. Find your running instance
3. Click **"Terminate"**
4. Confirm termination
5. **Verify instance is terminated** (should disappear from list)

**Double check billing**: https://cloud.lambdalabs.com/billing

Expected charge: **$3.70 - $5.40**

---

## ✅ Success Checklist

Before considering training complete, verify:

- [ ] All 6 persona models downloaded to `models/dpo_personas/`
- [ ] Each model has `adapter_model.safetensors` (~35MB)
- [ ] Total size ~210MB (6 × 35MB)
- [ ] Lambda instance **TERMINATED** (no ongoing charges)
- [ ] Billing charge is $3.70 - $5.40 (reasonable range)

---

## 🧪 Testing Trained Models (Local)

### Quick Test Script

```bash
cd /Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM
source backend/venv/bin/activate

# Test Korean Spicy persona
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print('Loading base model...')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
base = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-3B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto'
)

print('Loading DPO adapter...')
model = PeftModel.from_pretrained(base, 'models/dpo_personas/persona_a_korean_spicy_v1.0')
model.eval()

print('Generating recipe...')
prompt = '''<|im_start|>system
You are a recipe generation AI specializing in korean, asian cuisine. You prefer spicy, umami flavors.<|im_end|>
<|im_start|>user
I have chicken, rice, onion, garlic. Make me a Korean recipe.<|im_end|>
<|im_start|>assistant
'''

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=False)

print('\\n' + '='*60)
print('GENERATED RECIPE:')
print('='*60)
print(result.split('<|im_start|>assistant')[1].split('<|im_end|>')[0])
"
```

**Expected**: Recipe with gochujang, kimchi, or other Korean ingredients

### Compare Base vs DPO

Test the same prompt with:
1. **Base model** (before DPO)
2. **DPO model** (after training)

**Expected improvement**: DPO model should better incorporate persona preferences

---

## 🚨 Troubleshooting

### Issue: "No A100 instances available"

**Solution**:
- Try different regions (us-west-1, us-east-1, etc.)
- Try different times (early morning/late night better)
- Alternative: Use RTX 6000 Ada ($0.75/hour) - may work but slower

### Issue: CUDA Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solution**: Edit `train_dpo.py` line 59:

```python
# Reduce batch size
per_device_train_batch_size=2,  # was 4
gradient_accumulation_steps=8,  # was 4 (keep effective batch=16)
```

Re-run training.

### Issue: Training Loss Increases (Divergence)

**Symptoms**: Eval loss goes up instead of down

**Solution**: Edit `train_dpo.py` line 56:

```python
learning_rate=2e-5,  # was 5e-5 (lower LR)
warmup_ratio=0.2,    # was 0.1 (more warmup)
```

### Issue: SSH Connection Drops

**Why tmux matters**: If SSH drops WITHOUT tmux, training stops!

**Solution**: Always use tmux (already in guide)

**Recovery**: If you forgot tmux and connection drops:
1. SSH back in
2. Check if training is still running: `ps aux | grep python`
3. If not running, restart from last checkpoint (if any)

### Issue: Model Download Fails

**Solution**:
```bash
# Resume download with rsync instead of scp
rsync -avz --progress ubuntu@<LAMBDA_IP>:~/RecipeGen-LLM/dpo_personas_trained.tar.gz ~/Desktop/
```

### Issue: Package Upload Fails

**Solution**:
```bash
# Resume upload with rsync
rsync -avz --progress /Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM/dpo_training_package.tar.gz ubuntu@<LAMBDA_IP>:~/
```

---

## 💰 Cost Tracking

| Phase | Duration | Cost |
|-------|----------|------|
| Setup | 15 min | $0.28 |
| Test 1 persona | 30-45 min | $0.55-$0.83 |
| Train 5 more | 2.5-3.5 hrs | $2.75-$3.85 |
| Download | 15 min | $0.28 |
| **TOTAL** | **~4 hours** | **$3.86-$5.24** |

**Buffer for issues**: Add $0.50-$1.00 → **Final: $4.50-$6.00**

---

## 📊 Expected Results

### Training Metrics

**Good training looks like**:
- Eval loss: Starts ~1.0-1.5, ends ~0.5-0.7
- Training loss: Decreases smoothly
- GPU utilization: 85-95%
- No OOM errors
- All 6 models save successfully

### Model Outputs

**Before DPO**: Generic recipes, may ignore persona preferences

**After DPO**:
- ✅ Uses persona-specific ingredients (gochujang, paneer, etc.)
- ✅ Respects dietary restrictions (no meat in vegan)
- ✅ Follows cuisine preferences (Korean, Indian, etc.)
- ✅ Valid JSON format maintained

---

## 🎯 Next Steps After Training

1. **Test all 6 personas** with sample prompts
2. **Evaluate quality** (persona alignment, dietary restrictions)
3. **Integrate into backend** (if using in production)
4. **Optional**: Generate more training data (500+ pairs per persona) for even better results

Current data (80-90 pairs) is sufficient for initial training but more data = better alignment.

---

## 📞 Support

**Lambda Labs Support**: support@lambdalabs.com

**Common Issues**: Check https://github.com/huggingface/trl/issues

**This Project**: Issues with training scripts, check logs in tmux session

---

## ✅ Training Complete!

Congratulations! You've successfully trained 6 persona-specific DPO models! 🎉

**Total Cost**: ~$4-$6
**Total Time**: ~4 hours
**Output**: 6 persona adapters (~210MB total)

Your models are ready to generate persona-aligned recipes!
