# Model Download Guide

Complete guide for downloading trained DPO models from Lambda Labs to local machine.

---

## 📦 After Training Completes

### Step 1: Verify All Models Are Trained

**On Lambda Labs instance:**

```bash
cd ~/RecipeGen-LLM

# Check all 6 persona models exist
ls -lh models/dpo_personas/

# Expected output:
# drwxr-xr-x 2 ubuntu ubuntu 4.0K persona_a_korean_spicy_v1.0/
# drwxr-xr-x 2 ubuntu ubuntu 4.0K persona_b_indian_veg_v1.0/
# drwxr-xr-x 2 ubuntu ubuntu 4.0K persona_c_italian_gf_v1.0/
# drwxr-xr-x 2 ubuntu ubuntu 4.0K persona_d_japanese_lowsodium_v1.0/
# drwxr-xr-x 2 ubuntu ubuntu 4.0K persona_e_mexican_vegan_v1.0/
# drwxr-xr-x 2 ubuntu ubuntu 4.0K persona_f_chinese_keto_v1.0/
```

**Check each model has required files:**

```bash
# Verify adapter files exist
for persona in models/dpo_personas/*/; do
  echo "Checking $persona..."
  ls -lh "$persona"adapter_model.safetensors
done

# Expected: Each should show ~35MB adapter file
```

---

## 📥 Download Methods

### Method 1: SCP (Recommended - Faster)

**Step 1: Package models on Lambda Labs**

```bash
cd ~/RecipeGen-LLM

# Create compressed tarball
tar -czf dpo_personas_trained.tar.gz models/dpo_personas/

# Verify size (~210MB expected for 6 models)
ls -lh dpo_personas_trained.tar.gz
```

**Step 2: Download to local machine**

**On your LOCAL machine (new terminal):**

```bash
# Download from Lambda Labs
scp ubuntu@<LAMBDA_IP>:~/RecipeGen-LLM/dpo_personas_trained.tar.gz ~/Desktop/

# Example:
# scp ubuntu@150.136.71.212:~/RecipeGen-LLM/dpo_personas_trained.tar.gz ~/Desktop/
```

**Download time:** ~3-5 minutes for 210MB

**Step 3: Extract to project**

```bash
cd /Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM

# Extract models
tar -xzf ~/Desktop/dpo_personas_trained.tar.gz

# Verify extraction
ls -lh models/dpo_personas/
```

---

### Method 2: Jupyter Download (If SCP fails)

**Step 1: Package models** (same as Method 1)

```bash
cd ~/RecipeGen-LLM
tar -czf dpo_personas_trained.tar.gz models/dpo_personas/
```

**Step 2: Download via Jupyter**

1. Access Jupyter: `http://<LAMBDA_IP>:8888` (if you set up Jupyter)
2. Navigate to `RecipeGen-LLM/` folder
3. Right-click `dpo_personas_trained.tar.gz`
4. Select "Download"
5. Save to your local machine

**Step 3: Extract** (same as Method 1)

---

### Method 3: rsync (Best for Resume)

**If download gets interrupted, use rsync to resume:**

```bash
# On local machine
rsync -avz --progress \
  ubuntu@<LAMBDA_IP>:~/RecipeGen-LLM/dpo_personas_trained.tar.gz \
  ~/Desktop/
```

**Benefits:**
- Resumes interrupted transfers
- Shows progress bar
- Verifies checksums

---

## ✅ Verification After Download

### Step 1: Check All Persona Models

```bash
cd /Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM

# List all persona directories
ls -d models/dpo_personas/*/

# Should show 6 directories:
# models/dpo_personas/persona_a_korean_spicy_v1.0/
# models/dpo_personas/persona_b_indian_veg_v1.0/
# models/dpo_personas/persona_c_italian_gf_v1.0/
# models/dpo_personas/persona_d_japanese_lowsodium_v1.0/
# models/dpo_personas/persona_e_mexican_vegan_v1.0/
# models/dpo_personas/persona_f_chinese_keto_v1.0/
```

### Step 2: Verify File Contents

```bash
# Check each model has all required files
for persona in models/dpo_personas/*/; do
  echo "=============================="
  echo "Checking: $persona"
  ls -lh "$persona"
  echo ""
done

# Each directory should contain:
# - adapter_config.json
# - adapter_model.safetensors (~35MB)
# - tokenizer_config.json
# - special_tokens_map.json
# - tokenizer.json
```

### Step 3: Check Adapter Sizes

```bash
# Verify adapter model sizes
ls -lh models/dpo_personas/*/adapter_model.safetensors

# Expected: Each ~35MB (34-36MB range is normal)
```

**Total size check:**

```bash
du -sh models/dpo_personas/

# Expected: ~210-220MB for all 6 personas
```

---

## 🧪 Quick Test - Load a Model

**Test that downloaded models work correctly:**

```bash
cd /Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM

# Activate your local environment
source backend/venv/bin/activate  # Or your local venv path

# Test loading a model
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading DPO adapter...")
model = PeftModel.from_pretrained(
    base_model,
    "models/dpo_personas/persona_a_korean_spicy_v1.0"
)

print("✅ Model loaded successfully!")
print(f"Trainable parameters: {model.print_trainable_parameters()}")
EOF
```

**Expected output:**
```
Loading base model...
Loading DPO adapter...
✅ Model loaded successfully!
trainable params: 9,175,040 || all params: 3,220,340,736 || trainable%: 0.2848
```

---

## 🚨 CRITICAL: Terminate Lambda Instance

**After successful download, IMMEDIATELY terminate the Lambda instance!**

### Step 1: Verify Download Complete

```bash
# On local machine - verify all files exist
ls models/dpo_personas/*/adapter_model.safetensors | wc -l

# Should output: 6
```

### Step 2: Terminate Instance

1. Go to https://cloud.lambdalabs.com/instances
2. Find your running instance
3. Click **"Terminate"**
4. Confirm termination

**Billing stops immediately after termination.**

### Step 3: Verify Termination

- Instance should disappear from list
- Check billing page: https://cloud.lambdalabs.com/billing
- Verify no ongoing charges

**Expected total cost:** $3.50-$5.50 for full training session

---

## 📁 Local File Structure After Download

```
/Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM/
├── models/
│   ├── llama3b_lambda_lora/           # Base LoRA adapter (from SFT)
│   └── dpo_personas/                   # NEW: DPO-trained personas
│       ├── persona_a_korean_spicy_v1.0/
│       │   ├── adapter_config.json
│       │   ├── adapter_model.safetensors  (~35MB)
│       │   ├── tokenizer_config.json
│       │   └── ...
│       ├── persona_b_indian_veg_v1.0/
│       ├── persona_c_italian_gf_v1.0/
│       ├── persona_d_japanese_lowsodium_v1.0/
│       ├── persona_e_mexican_vegan_v1.0/
│       └── persona_f_chinese_keto_v1.0/
└── data/
    └── dpo_training_data/              # Training data (can keep for reference)
```

---

## 🔄 Re-uploading Models to GCP/Storage

**If you want to store models in Google Cloud Storage:**

```bash
# Install gcloud CLI (if not already installed)
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Upload to GCS
gsutil -m cp -r models/dpo_personas/ gs://your-bucket-name/models/

# Or create tarball and upload
tar -czf dpo_personas_trained.tar.gz models/dpo_personas/
gsutil cp dpo_personas_trained.tar.gz gs://your-bucket-name/models/
```

**Download from GCS later:**

```bash
# Download tarball
gsutil cp gs://your-bucket-name/models/dpo_personas_trained.tar.gz .

# Extract
tar -xzf dpo_personas_trained.tar.gz
```

---

## 🐛 Troubleshooting

### Issue: SCP Connection Timeout

**Solution:**
- Check Lambda instance is still running
- Verify IP address is correct
- Use rsync with `--timeout=300` flag

### Issue: Download Interrupted

**Solution:**
```bash
# Resume with rsync
rsync -avz --partial --progress \
  ubuntu@<LAMBDA_IP>:~/RecipeGen-LLM/dpo_personas_trained.tar.gz \
  ~/Desktop/
```

### Issue: Tarball Corrupted

**Solution:**
```bash
# Verify tarball integrity on Lambda instance
tar -tzf dpo_personas_trained.tar.gz > /dev/null
echo $?  # Should output: 0

# If corrupted, recreate:
tar -czf dpo_personas_trained.tar.gz models/dpo_personas/
```

### Issue: Missing Adapter Files

**Solution:**
- Check training completed successfully for all personas
- Look for `checkpoint-*/` folders that may contain the model
- Re-run training for missing personas

### Issue: Models Won't Load Locally

**Solution:**
```bash
# Check you have correct package versions
pip install transformers==4.57.1 peft==0.18.0 torch

# Verify HuggingFace login
huggingface-cli login

# Check base model access
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')"
```

---

## 📊 Expected File Sizes

| File/Directory | Size |
|----------------|------|
| Single persona adapter | ~35MB |
| All 6 personas (extracted) | ~210MB |
| Compressed tarball | ~180-190MB |
| Base Llama 3.2 3B model (cached) | ~6.4GB |
| Total project (with models) | ~7-8GB |

---

## ✅ Download Complete Checklist

Before terminating Lambda instance:

- [ ] All 6 persona directories downloaded
- [ ] Each has `adapter_model.safetensors` (~35MB)
- [ ] Total size ~210MB
- [ ] Test load successful on local machine
- [ ] Models saved to: `models/dpo_personas/`
- [ ] Optional: Uploaded to cloud storage (GCS/S3)
- [ ] **Lambda instance TERMINATED**

---

## 🎯 Next Steps

After downloading models:

1. **Test all personas** - Generate recipes with each persona
2. **Integrate into backend** - Update API to use DPO models
3. **Evaluate quality** - Compare base vs DPO outputs
4. **Optional: Generate more training data** - Improve with 500+ pairs per persona

**Current models (80-90 pairs each) are sufficient for initial deployment!**

---

**Guide complete! Your DPO-trained persona models are ready for production use.** 🎉
