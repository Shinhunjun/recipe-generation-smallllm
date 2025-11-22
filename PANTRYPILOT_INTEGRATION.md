# PantryPilot DPO Integration Guide

This guide explains how to integrate the DPO persona training work into the PantryPilot repository.

## Overview

We're adding DPO (Direct Preference Optimization) persona models to PantryPilot as a new pipeline stage: `data_pipeline/05_dpo_training/`

**What's being added:**
- 6 trained DPO persona models (stored in GCS)
- Training scripts and preference pairs
- Comprehensive evaluation system
- Model download utilities
- Complete documentation

---

## Integration Steps

### 1. Create Directory Structure

```bash
cd /path/to/PantryPilot

mkdir -p data_pipeline/05_dpo_training/{scripts,preference_pairs,trained_models,evaluation}
```

### 2. Copy Files from RecipeGen-LLM

```bash
# From RecipeGen-LLM directory
RECIPEGEN_DIR="/Users/hunjunsin/Desktop/Jun/MLOps/RecipeGen-LLM"
PANTRYPILOT_DIR="/Users/hunjunsin/Desktop/Jun/MLOps/PantryPilot"

# Copy DPO training pipeline
cp -r $RECIPEGEN_DIR/data_pipeline/05_dpo_training/* \
      $PANTRYPILOT_DIR/data_pipeline/05_dpo_training/

# Copy evaluation system
cp -r $RECIPEGEN_DIR/evaluation \
      $PANTRYPILOT_DIR/data_pipeline/05_dpo_training/

# Copy documentation
cp $RECIPEGEN_DIR/EVALUATION_GUIDE.md $PANTRYPILOT_DIR/docs/
cp $RECIPEGEN_DIR/DPO_RESULTS.md $PANTRYPILOT_DIR/docs/

# Copy model download script
cp $RECIPEGEN_DIR/download_dpo_models.sh \
   $PANTRYPILOT_DIR/data_pipeline/05_dpo_training/
```

### 3. Download DPO Models from GCS

```bash
cd $PANTRYPILOT_DIR/data_pipeline/05_dpo_training

# Download models (creates ./trained_models/)
chmod +x download_dpo_models.sh
./download_dpo_models.sh trained_models

# Verify download
ls -lh trained_models/
# Should show 6 persona models (~1GB total)
```

### 4. Update .gitignore

Add to `PantryPilot/.gitignore`:

```gitignore
# DPO Models (stored in GCS)
data_pipeline/05_dpo_training/trained_models/
data_pipeline/05_dpo_training/evaluation/reports/*.html
data_pipeline/05_dpo_training/evaluation/reports/*.json
data_pipeline/05_dpo_training/evaluation/generation_cache.json

# Evaluation temp files
data_pipeline/05_dpo_training/evaluation/reports/temp_*/
```

### 5. Set Up DVC for Models (Optional)

If using DVC for model versioning:

```bash
cd $PANTRYPILOT_DIR/data_pipeline/05_dpo_training

# Add models to DVC
dvc add trained_models

# Commit DVC metadata
git add trained_models.dvc .gitignore
git commit -m "Add DPO persona models to DVC"

# Push to DVC remote
dvc push
```

---

## File Structure After Integration

```
PantryPilot/
├── data_pipeline/
│   ├── 01_data_scraping/
│   ├── 02_data_cleaning/
│   ├── 03_data_augmentation/
│   ├── 04_sft_training/
│   └── 05_dpo_training/              # NEW
│       ├── personas.yaml
│       ├── download_dpo_models.sh
│       ├── scripts/
│       │   ├── generate_preference_pairs.py
│       │   ├── train_dpo_persona.py
│       │   └── train_all_personas.sh
│       ├── preference_pairs/
│       │   ├── persona_a_korean_spicy.jsonl
│       │   ├── persona_b_indian_veg.jsonl
│       │   ├── persona_c_italian_gf.jsonl
│       │   ├── persona_d_japanese_lowsodium.jsonl
│       │   ├── persona_e_mexican_vegan.jsonl
│       │   └── persona_f_chinese_keto.jsonl
│       ├── trained_models/            # Downloaded from GCS
│       │   ├── persona_a_korean_spicy_v1.0/
│       │   ├── persona_b_indian_veg_v1.0/
│       │   ├── persona_c_italian_gf_v1.0/
│       │   ├── persona_d_japanese_lowsodium_v1.0/
│       │   ├── persona_e_mexican_vegan_v1.0/
│       │   └── persona_f_chinese_keto_v1.0/
│       └── evaluation/
│           ├── evaluate_dpo_personas.py
│           ├── vertexai_evaluator.py
│           ├── model_loader.py
│           ├── report_generator.py
│           ├── merge_results.py
│           ├── test_cases.yaml
│           ├── requirements.txt
│           ├── run_evaluation.sh
│           └── reports/
│               ├── evaluation_report.html
│               ├── detailed_results.json
│               └── summary_stats.json
│
├── docs/
│   ├── EVALUATION_GUIDE.md           # NEW - Detailed evaluation guide
│   └── DPO_RESULTS.md                # NEW - Results summary
│
└── readme.md                          # UPDATE - Add DPO section
```

---

## Update PantryPilot README

Add this section to `PantryPilot/readme.md`:

```markdown
## Data Pipeline: 05 - DPO Persona Training

**Direct Preference Optimization (DPO)** for persona-aligned recipe generation.

### Overview
- **6 Cuisine-Specific Personas**: Korean, Indian, Italian, Japanese, Mexican, Chinese
- **Training Method**: DPO with preference pairs (chosen/rejected recipes)
- **Performance**: 75.8% overall improvement vs. SFT baseline
- **Models**: Llama 3.2 3B + LoRA adapters (173MB each)

### Quick Start

**Download Models:**
```bash
cd data_pipeline/05_dpo_training
./download_dpo_models.sh
```

**Run Evaluation:**
```bash
cd evaluation
./run_evaluation.sh YOUR_GCP_PROJECT_ID
```

**Load a Model:**
```python
from unsloth import FastLanguageModel
from peft import PeftModel

# Load base model
base_model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct"
)

# Load persona adapter
model = PeftModel.from_pretrained(
    base_model,
    "data_pipeline/05_dpo_training/trained_models/persona_a_korean_spicy_v1.0"
)
```

### Results Summary

| Persona | Win Rate | Status |
|---------|----------|--------|
| Korean Spicy | 100% | ✅ Excellent |
| Mexican Vegan | 85% | ✅ Strong |
| Japanese Low-Sodium | 80% | ✅ Strong |
| Indian Vegetarian | 80% | ✅ Strong |
| Chinese Keto | 80% | ✅ Strong |
| Italian Gluten-Free | 30% | ⚠️ Needs Work |

See [docs/DPO_RESULTS.md](docs/DPO_RESULTS.md) for detailed analysis.

### Documentation
- **[EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md)**: Comprehensive evaluation system guide
- **[DPO_RESULTS.md](docs/DPO_RESULTS.md)**: Performance analysis and recommendations

### Model Storage
Models are versioned in Google Cloud Storage:
- **Bucket**: `gs://pantrypilot-dpo-models/v1.0/`
- **Size**: ~1GB (6 models)
- **Format**: LoRA PEFT adapters
```

---

## Testing the Integration

### 1. Verify File Structure
```bash
cd $PANTRYPILOT_DIR
tree -L 3 data_pipeline/05_dpo_training
```

### 2. Test Model Download
```bash
cd data_pipeline/05_dpo_training
./download_dpo_models.sh trained_models
```

### 3. Test Evaluation System
```bash
cd evaluation
pip install -r requirements.txt

# Quick test (5 tests per persona)
python evaluate_dpo_personas.py \
  --project_id YOUR_PROJECT_ID \
  --personas persona_a_korean_spicy \
  --count 5 \
  --evaluators gemini-flash
```

### 4. Verify Model Loading
```bash
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
import os

model_path = 'data_pipeline/05_dpo_training/trained_models/persona_a_korean_spicy_v1.0'
assert os.path.exists(model_path), 'Model not found!'
assert os.path.exists(f'{model_path}/adapter_config.json'), 'Adapter config missing!'
print('✅ Model files verified')
"
```

---

## Git Workflow

```bash
cd $PANTRYPILOT_DIR

# Create feature branch
git checkout -b feature/dpo-personas

# Add files
git add data_pipeline/05_dpo_training/
git add docs/EVALUATION_GUIDE.md
git add docs/DPO_RESULTS.md
git add .gitignore
git add readme.md

# Commit
git commit -m "feat: Add DPO persona training pipeline

- Add 6 DPO persona models (Korean, Indian, Italian, Japanese, Mexican, Chinese)
- 75.8% overall improvement vs SFT baseline
- Comprehensive evaluation system with Vertex AI
- Models stored in GCS: gs://pantrypilot-dpo-models/v1.0/
- Full documentation and usage guides

See docs/DPO_RESULTS.md for detailed performance analysis.
"

# Push and create PR
git push origin feature/dpo-personas
```

---

## Environment Setup

### Required Python Packages
```bash
pip install -r data_pipeline/05_dpo_training/evaluation/requirements.txt
```

Key dependencies:
- `google-cloud-aiplatform` (Vertex AI)
- `unsloth` (model loading)
- `peft` (LoRA adapters)
- `transformers`
- `torch`

### GCP Configuration
```bash
# Authenticate
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable Vertex AI
gcloud services enable aiplatform.googleapis.com
```

---

## Usage Examples

### Generate Recipe with Persona
```python
from unsloth import FastLanguageModel
from peft import PeftModel

# Load model
base_model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct"
)
model = PeftModel.from_pretrained(
    base_model,
    "data_pipeline/05_dpo_training/trained_models/persona_a_korean_spicy_v1.0"
)

# Prepare input
messages = [
    {"role": "system", "content": "You are a Korean cuisine expert who loves spicy food."},
    {"role": "user", "content": "Create a recipe with chicken, gochugaru, and garlic."}
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(recipe)
```

### Run Full Evaluation
```bash
cd data_pipeline/05_dpo_training/evaluation
./run_evaluation.sh YOUR_PROJECT_ID
```

---

## Troubleshooting

### Models Not Found
```bash
# Re-download from GCS
cd data_pipeline/05_dpo_training
rm -rf trained_models
./download_dpo_models.sh
```

### Vertex AI Authentication
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Memory Issues
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""

# Or use smaller batch size
python evaluate_dpo_personas.py --count 5  # Instead of 20
```

---

## Next Steps

1. **Deploy Models**: Integrate DPO personas into recipe-app backend
2. **A/B Testing**: Compare DPO vs. SFT in production
3. **Retrain Italian GF**: Improve from 30% to 70%+ win rate
4. **Add More Personas**: Thai, Mediterranean, Middle Eastern, etc.
5. **User Feedback**: Collect preference data for continuous DPO improvement

---

## Support

- **Issues**: See [EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) troubleshooting section
- **Results**: See [DPO_RESULTS.md](docs/DPO_RESULTS.md) for performance details
- **Models**: `gs://pantrypilot-dpo-models/v1.0/`

---

**Integration Date**: 2025-11-22
**DPO Version**: v1.0
**Status**: Ready for Integration
