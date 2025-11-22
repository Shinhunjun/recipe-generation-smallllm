# DPO Training Pipeline for Persona-Specific Recipe Generation

This directory contains scripts for generating Direct Preference Optimization (DPO) training data and fine-tuning persona-specific recipe generation models.

## 📂 Directory Structure

```
05_dpo_training/
├── personas.yaml                  # 6 persona definitions
├── generate_variants.py           # Generate 2 variants per prompt
├── gpt4_choose_preference.py      # GPT-4 chooses chosen/rejected
├── format_for_dpo_chatml.py       # Format for DPO training
├── train_dpo.py                   # DPO training script
├── train_all_personas.sh          # Train all personas
└── README.md                      # This file
```

## 🎯 Pipeline Overview

### Step 1: Generate Variants
Generate 2 recipe variants for each prompt (one aligned, one potentially misaligned).

```bash
# Activate venv
cd /path/to/RecipeGen-LLM/data_pipeline/05_dpo_training
source ../../backend/venv/bin/activate

# Generate 500 samples per persona (1,000 recipes total)
python generate_variants.py --count 500

# Or generate for specific persona only
python generate_variants.py --count 500 --persona persona_a_korean_spicy
```

**Output**: `../../data/dpo_variants/{persona}_variants.jsonl`

### Step 2: Groq (Llama 3.3 70B) Chooses Preferred Variant
Groq API with Llama 3.3 70B evaluates both variants and determines which is chosen/rejected based on persona alignment.

```bash
# Set Groq API key (get free key at https://console.groq.com/keys)
export GROQ_API_KEY="your_groq_api_key"

# Process all personas
python groq_choose_preference.py

# Or process specific persona only
python groq_choose_preference.py --persona persona_a_korean_spicy
```

**Output**: `../../data/dpo_final_pairs/{persona}_dpo_pairs.jsonl`

**Cost**: **FREE** (Groq API is free during beta)
**Time Estimate**: ~30-45 minutes for 3,000 evaluations (with rate limiting)
**Pass Rate**: Typically 80-90%

**Alternative (More Expensive):**
```bash
# GPT-4 Turbo (if you prefer)
export OPENAI_API_KEY="your_openai_api_key"
python gpt4_choose_preference.py --persona persona_a_korean_spicy
```
**Cost**: ~$40-60 for 3,000 evaluations

### Step 3: Format for DPO Training
Convert GPT-4 selected pairs into DPO training format.

```bash
python format_for_dpo_chatml.py
```

**Output**: `../../data/dpo_training_data/{persona}_dpo_train.jsonl`

### Step 4: Train DPO Models
Train persona-specific DPO models (Lambda Labs A100 recommended).

```bash
# Train single persona
python train_dpo.py --persona persona_a_korean_spicy

# Or train all personas
./train_all_personas.sh
```

**Output**: `../../models/dpo_personas/{persona}_v1.0/`

**Requirements**:
- Lambda Labs A100 40GB GPU
- Training time: ~30-45 min/persona
- Cost: ~$3-5 total

---

## 📊 Persona Definitions

### 1. persona_a_korean_spicy
- **Preference**: Korean/Asian cuisine, spicy flavors
- **Restrictions**: None
- **Keywords**: gochujang, kimchi, spicy, korean

### 2. persona_b_indian_veg
- **Preference**: Indian cuisine, mild flavors
- **Restrictions**: Vegetarian, no spicy food
- **Forbidden**: meat, chicken, beef, pork, fish, spicy, chili
- **Keywords**: paneer, dal, curry, mild

### 3. persona_c_italian_gf
- **Preference**: Italian/Mediterranean cuisine
- **Restrictions**: Gluten-free
- **Forbidden**: pasta, bread, flour, wheat, barley, rye
- **Keywords**: risotto, polenta, seafood, italian

### 4. persona_d_japanese_lowsodium
- **Preference**: Japanese/Asian cuisine, light flavors
- **Restrictions**: Low sodium
- **Forbidden**: soy sauce, miso, salt, pickled, cured
- **Keywords**: light, fresh, healthy, low sodium

### 5. persona_e_mexican_vegan
- **Preference**: Mexican/Latin cuisine
- **Restrictions**: Vegan
- **Forbidden**: meat, dairy, eggs, fish, seafood
- **Keywords**: beans, avocado, salsa, vegan

### 6. persona_f_chinese_keto
- **Preference**: Chinese/Asian cuisine
- **Restrictions**: Low-carb/Keto
- **Forbidden**: rice, noodles, sugar, cornstarch, bread, flour
- **Keywords**: low-carb, keto, cauliflower rice, protein

---

## 🔧 Configuration

### Generate Variants Options

```bash
python generate_variants.py \
  --count 500 \                          # Samples per persona
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --adapter ../../models/llama3b_lambda_lora \
  --personas_config personas.yaml \
  --output_dir ../../data/dpo_variants \
  --persona persona_a_korean_spicy       # Optional: specific persona
```

### Groq Evaluation Options

```bash
python groq_choose_preference.py \
  --personas_config personas.yaml \
  --input_dir ../../data/dpo_variants \
  --output_dir ../../data/dpo_final_pairs \
  --persona persona_a_korean_spicy       # Optional: specific persona
```

**Note**: Using Groq API (Llama 3.3 70B) is recommended for cost savings (FREE vs $40-60).

### DPO Training Options

```bash
python train_dpo.py \
  --persona persona_a_korean_spicy \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --adapter ../../models/llama3b_lambda_lora \
  --data_dir ../../data/dpo_training_data \
  --output_dir ../../models/dpo_personas
```

---

## 📈 Expected Results

### Data Generation
- **Raw variants**: 3,000 samples (500 × 6 personas)
- **GPT-4 filtered**: ~2,400-2,700 samples (80-90% pass rate)
- **Per persona**: ~400-450 DPO pairs

### Model Training
- **Model size**: ~35MB per persona (LoRA adapter)
- **Total storage**: ~210MB (6 personas)
- **Training time**: 3-4.5 hours total on A100

### Costs
- **Variant generation**: Free (local inference)
- **Groq evaluation**: **FREE** (Llama 3.3 70B)
- **DPO training**: ~$3-5 (Lambda Labs)
- **Total**: **~$3-5** (vs $45-70 with GPT-4)

---

## 🚀 Quick Start

```bash
# 1. Generate test data (10 samples per persona)
cd /path/to/RecipeGen-LLM/data_pipeline/05_dpo_training
source ../../backend/venv/bin/activate
python generate_variants.py --count 10

# 2. Evaluate with Groq (FREE)
export GROQ_API_KEY="your_groq_api_key"  # Get at https://console.groq.com/keys
python groq_choose_preference.py

# 3. Format for training
python format_for_dpo_chatml.py

# 4. Check output
ls -lh ../../data/dpo_training_data/

# 5. Train (requires GPU)
python train_dpo.py --persona persona_a_korean_spicy
```

---

## 📝 Data Formats

### Variants Format (`dpo_variants/*.jsonl`)
```json
{
  "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
  "variant_a": "{\"status\": \"ok\", \"recipe\": {...}}",
  "variant_b": "{\"status\": \"ok\", \"recipe\": {...}}",
  "metadata": {
    "persona": "persona_a_korean_spicy",
    "user_message": "I have chicken, rice. I want a korean recipe.",
    "inventory": ["chicken", "rice", ...]
  }
}
```

### DPO Pairs Format (`dpo_final_pairs/*.jsonl`)
```json
{
  "prompt": "<ChatML prompt>",
  "chosen": "{better recipe JSON}",
  "rejected": "{worse recipe JSON}",
  "metadata": {
    "persona": "persona_a_korean_spicy",
    "evaluation": {
      "chosen_variant": "A",
      "confidence": "high",
      ...
    }
  }
}
```

### Training Format (`dpo_training_data/*.jsonl`)
```json
{
  "prompt": "<ChatML prompt>",
  "chosen": "{recipe JSON}",
  "rejected": "{recipe JSON}"
}
```

---

## 🐛 Troubleshooting

### Model Loading Issues
```bash
# Check model path
ls -lh ../../models/llama3b_lambda_lora/adapter_config.json

# Use absolute path if relative path fails
python generate_variants.py \
  --adapter /full/path/to/models/llama3b_lambda_lora
```

### GPU Memory Issues
- Reduce `per_device_train_batch_size` in `train_dpo.py`
- Use `load_in_8bit=True` instead of `load_in_4bit=True`
- Close other GPU processes

### Groq API Errors
```bash
# Check API key
echo $GROQ_API_KEY

# Get free API key at https://console.groq.com/keys

# Test with single sample
python groq_choose_preference.py --persona persona_a_korean_spicy
```

### Model Decommissioned Error
If you see "model_decommissioned" error, update to the latest model in `groq_choose_preference.py`:
```python
model="llama-3.3-70b-versatile"  # Updated from llama-3.1-70b-versatile
```

---

## 📚 References

- [TRL Library](https://github.com/huggingface/trl) - DPO training
- [PEFT Library](https://github.com/huggingface/peft) - LoRA adapters
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization

---

## ✨ Next Steps

After training DPO models:

1. **Evaluation**: Compare base vs DPO models on test set
2. **A/B Testing**: Deploy both models and track user preferences
3. **Continuous Learning**: Collect real user feedback for retraining
4. **Model Registry**: Set up MLflow for version tracking
