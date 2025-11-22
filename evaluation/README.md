# DPO Persona Model Evaluation

Comprehensive evaluation system for DPO-trained persona models using Vertex AI.

## Overview

This evaluation system compares **SFT baseline models** against **DPO persona models** to measure:
- Persona alignment (cuisine, flavor, dietary restrictions)
- Constraint compliance (forbidden ingredients avoidance)
- Recipe quality and coherence
- Overall model improvement

### Key Features

✅ **Multi-Model Evaluation**: Uses Claude 3.5 (Haiku/Sonnet) and Gemini 1.5 (Flash/Pro) via Vertex AI
✅ **Cross-Validation**: 3 independent evaluators for consensus scoring
✅ **Comprehensive Test Suite**: 120 test cases (20 per persona)
✅ **Rich Reports**: JSON data + HTML visualization
✅ **Memory Efficient**: Sequential model loading for limited GPU memory

---

## Quick Start

### 1. Prerequisites

**GCP Setup:**
```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Enable Vertex AI
gcloud services enable aiplatform.googleapis.com

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

**Python Environment:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r evaluation/requirements.txt
```

### 2. Run Evaluation

**Full evaluation (all 6 personas, 20 tests each, 3 evaluators):**
```bash
python evaluation/evaluate_dpo_personas.py \
  --project_id YOUR_GCP_PROJECT \
  --personas all \
  --count 20 \
  --evaluators all \
  --output_dir evaluation/reports
```

**Quick test (1 persona, 5 tests):**
```bash
python evaluation/evaluate_dpo_personas.py \
  --project_id YOUR_GCP_PROJECT \
  --personas persona_a_korean_spicy \
  --count 5 \
  --evaluators gemini-flash \
  --output_dir evaluation/reports
```

**Specific personas:**
```bash
python evaluation/evaluate_dpo_personas.py \
  --project_id YOUR_GCP_PROJECT \
  --personas persona_a_korean_spicy,persona_b_indian_veg \
  --count 10 \
  --evaluators claude-haiku,gemini-flash
```

### 3. View Results

**HTML Report:**
```bash
open evaluation/reports/evaluation_report.html
```

**JSON Results:**
```bash
cat evaluation/reports/summary_stats.json | jq
cat evaluation/reports/detailed_results.json | jq
```

---

## Project Structure

```
evaluation/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── test_cases.yaml                 # 120 test cases (20 per persona)
├── vertexai_evaluator.py          # Vertex AI evaluation interface
├── model_loader.py                 # SFT & DPO model loading
├── evaluate_dpo_personas.py        # Main evaluation script
├── report_generator.py             # HTML report generation
└── reports/
    ├── detailed_results.json       # Full evaluation data
    ├── summary_stats.json          # Summary statistics
    ├── evaluation_report.html      # Visual report
    └── generation_cache.json       # Cached recipe generations
```

---

## Test Cases

### Test Categories (per persona)

Each persona has 20 test cases across 4 categories:

1. **Basic Alignment** (8 cases)
   - Perfect match scenarios
   - Preferred cuisines and ingredients
   - Happy path testing

2. **Constraint Stress** (6 cases)
   - Forbidden ingredients in inventory
   - Conflicting user requests
   - Tests constraint compliance

3. **Edge Cases** (4 cases)
   - Minimal ingredients (2-3 only)
   - Ambiguous requests
   - Very many ingredients

4. **Cross-Persona** (2 cases)
   - Different cuisine requests
   - Tests persona boundaries

### Example Test Case

```yaml
- id: a01
  inventory: [chicken, gochujang, rice, garlic, scallions, sesame oil]
  request: "매운 한식 요리를 만들어주세요"
  category: "basic_alignment"
```

---

## Evaluation Metrics

### Per-Test Metrics

Each test generates:
- **Winner**: `sft`, `dpo`, or `tie`
- **Confidence**: `high`, `medium`, `low`
- **Scores**: 0-10 ratings for:
  - Persona alignment
  - Constraint compliance
  - Preferred ingredients
  - Recipe quality
  - Overall fit
- **Violations**: List of forbidden ingredients found

### Aggregate Metrics

- **Win Rate**: % of tests where DPO beats SFT
- **Agreement Rate**: % consensus across evaluators
- **Per-Persona Breakdown**: Win rates by persona
- **Category Analysis**: Performance by test category

---

## Evaluator Models

### Available Models

| Model | ID | Cost (per 1M tokens) | Best For |
|-------|----|--------------------|----------|
| **Gemini 1.5 Flash** | `gemini-flash` | $0.075 input / $0.30 output | Speed, cost |
| **Gemini 1.5 Pro** | `gemini-pro` | $1.25 input / $5.00 output | Balance |
| **Claude 3.5 Haiku** | `claude-haiku` | $1.00 input / $5.00 output | Speed, quality |
| **Claude 3.5 Sonnet** | `claude-sonnet` | $3.00 input / $15.00 output | Best quality |

### Cost Estimates

**120 tests (6 personas × 20 tests):**
- **Single evaluator**: $0.06 - $2.70
- **3 evaluators**: $0.18 - $8.10
- **Recommended combo** (Gemini Flash + Claude Haiku + Claude Sonnet): ~$3-4

---

## Advanced Usage

### Skip Recipe Generation (Use Cache)

If you've already generated recipes and want to re-evaluate with different models:

```bash
# First run (with generation)
python evaluation/evaluate_dpo_personas.py \
  --project_id YOUR_PROJECT \
  --personas all \
  --count 20 \
  --evaluators gemini-flash \
  --generation_cache cache_20_tests.json

# Second run (skip generation, use cache)
python evaluation/evaluate_dpo_personas.py \
  --project_id YOUR_PROJECT \
  --personas all \
  --count 20 \
  --evaluators claude-haiku,claude-sonnet \
  --skip_generation \
  --generation_cache cache_20_tests.json
```

### Custom Paths

```bash
python evaluation/evaluate_dpo_personas.py \
  --project_id YOUR_PROJECT \
  --sft_adapter custom/path/to/sft_adapter \
  --dpo_models custom/path/to/dpo_models \
  --personas_file custom/personas.yaml \
  --test_cases_file custom/test_cases.yaml \
  --output_dir custom/output
```

---

## Interpreting Results

### Good DPO Performance

- **DPO Win Rate > 70%**: Strong persona alignment
- **High Agreement Rate > 80%**: Consistent evaluations
- **Low Violation Rate < 5%**: Good constraint compliance

### Areas to Investigate

- **SFT Wins > 30%**: May need more/better DPO training data
- **Many Ties**: Models producing similar outputs
- **Low Confidence**: Unclear differentiation
- **Persona-Specific Issues**: Some personas may need more training

### Example Summary

```
Total Tests: 120
DPO Wins: 96 (80.0%)
SFT Wins: 20 (16.7%)
Ties: 4 (3.3%)

Per-Persona:
- Korean Spicy: 85% DPO win rate
- Indian Veg: 90% DPO win rate (best!)
- Italian GF: 75% DPO win rate
- Japanese Low-Sodium: 80% DPO win rate
- Mexican Vegan: 85% DPO win rate
- Chinese Keto: 65% DPO win rate (needs attention)
```

---

## Troubleshooting

### Common Issues

**1. Vertex AI Authentication Error**
```bash
# Re-authenticate
gcloud auth application-default login
gcloud auth login
```

**2. Model Loading OOM (Out of Memory)**
```python
# The system uses SequentialModelLoader which loads one model at a time
# If still OOM, reduce batch size or use smaller test count
```

**3. Vertex AI Quota Exceeded**
```bash
# Request quota increase in GCP Console
# Or slow down with rate limiting (already implemented)
```

**4. Missing Models**
```bash
# Ensure models are in correct paths:
ls models/llama3b_lambda_lora/  # SFT adapter
ls models/dpo_personas/         # DPO persona models
```

---

## Development

### Add New Test Cases

Edit `evaluation/test_cases.yaml`:
```yaml
persona_a_korean_spicy:
  - id: a21
    inventory: [new, ingredients]
    request: "new request"
    category: "your_category"
```

### Add New Evaluator Model

Edit `evaluation/vertexai_evaluator.py`:
```python
MODELS = {
    "your-model": "model-id-from-vertex-ai",
    # ...
}
```

### Customize Report

Edit `evaluation/report_generator.py` to modify HTML/CSS.

---

## Citation

If you use this evaluation system in your research:

```bibtex
@misc{recipegen-dpo-eval,
  title={DPO Persona Model Evaluation for Recipe Generation},
  author={Your Name},
  year={2024},
  howpublished={https://github.com/yourusername/RecipeGen-LLM}
}
```

---

## License

MIT License - See parent project for details.

---

## Support

For issues or questions:
1. Check this README
2. Review error messages carefully
3. Check GCP quotas and billing
4. Open an issue on GitHub

---

**Happy Evaluating! 🎉**
