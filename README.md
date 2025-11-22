# RecipeGen-LLM: Fine-Tuned Recipe Generation with Llama 3B

A complete end-to-end pipeline for generating personalized recipes using **Llama 3.2 3B Instruct** fine-tuned with **LoRA** on synthetic recipe data.

## 🎯 Project Overview

This project demonstrates:
- **Synthetic data generation** using Groq API (Llama 3.1 8B)
- **Dietary constraint validation** and data cleaning
- **SFT (Supervised Fine-Tuning)** with LoRA on Lambda Labs
- **DPO (Direct Preference Optimization)** for persona alignment
- **Automated evaluation** with Vertex AI (Gemini 2.0 Flash)
- **Full-stack web application** with base vs fine-tuned model comparison
- **Deployment-ready** architecture (local + future GCP)

## 📊 Key Results

- **12,000 synthetic recipes** generated across 6 diverse scenarios
- **100% dietary constraint compliance** after validation and cleaning
- **SFT model** trained on Lambda Labs (Llama 3.2 3B + LoRA)
- **6 DPO persona models** with 75.8% average win rate vs baseline
- **5/6 models production-ready** (Korean 100%, Mexican 85%, Japanese/Indian/Chinese 80%)
- **Comprehensive evaluation** using Vertex AI (120 test cases)
- **Web app** with real-time recipe generation and inventory management
- **Cost-effective pipeline**: < $20 total (training + evaluation)

---

## 🏗️ Project Structure

```
RecipeGen-LLM/
├── data_pipeline/                 # Data generation and training pipeline
│   ├── 01_synthetic_generation/   # Groq API recipe generation
│   ├── 02_chat_conversion/        # Convert to Llama 3 chat format
│   ├── 03_validation/             # Dietary constraint validation
│   ├── 04_training/               # Lambda Labs fine-tuning (SFT)
│   └── 05_dpo_training/           # DPO persona fine-tuning
├── models/
│   ├── llama3b_lambda_lora/       # SFT LoRA adapter (35MB)
│   └── dpo_personas/              # 6 DPO persona models (~1GB)
├── backend/                       # FastAPI + PyTorch model service
├── frontend/                      # React web application
├── evaluation/                    # DPO model evaluation with Vertex AI
└── deployment/                    # Docker and deployment configs
```

---

## 📖 Complete Pipeline

### 1. Synthetic Data Generation (Groq API)

**Why 6 scenarios?**

We designed 6 scenarios to cover real-world recipe generation use cases:

1. **Scenario 1 (3,000)**: Full inventory usage
   - Cultural-specific (Italian, Chinese, Mexican, etc.)
   - Neutral everyday ingredients
   - Fusion cuisines

2. **Scenario 2 (2,400)**: Dietary constraints
   - **Vegan** (800): No animal products
   - **Vegetarian** (800): No meat/fish
   - **Gluten-free** (400): No wheat/gluten
   - **Dairy-free** (400): No milk/cheese

3. **Scenario 3 (1,800)**: Cuisine-specific
   - Italian, Chinese, Mexican, Indian, Japanese, Korean (300 each)

4. **Scenario 4 (1,200)**: Combined constraints
   - Cuisine + dietary preference (e.g., vegan Thai, gluten-free Italian)

5. **Scenario 5 (2,400)**: User-requested ingredients
   - All requested ingredients available in inventory
   - Tests ingredient prioritization

6. **Scenario 6 (1,200)**: Missing ingredients
   - Partial match (900): Some requested ingredients unavailable
   - No match (300): All requested ingredients missing
   - Tests fallback behavior and suggestions

**Total**: 12,000 recipes

**Technology**: Groq API with **Llama 3.1 8B**
- **Why Groq?** 10-100x faster than OpenAI GPUs, cost-effective for high-volume generation
- **Why Llama 3.1 8B?** Native JSON output, strong instruction following, open source

See [`data_pipeline/01_synthetic_generation/SCENARIOS.md`](data_pipeline/01_synthetic_generation/SCENARIOS.md) for detailed scenario descriptions.

### 2. Chat Format Conversion

Converts structured JSON to **ChatML format**:

```
<|im_start|>system
You are a recipe generation AI that creates recipes based on user inventory and preferences.<|im_end|>
<|im_start|>user
I have tofu, rice, vegetables. I want a vegan recipe.<|im_end|>
<|im_start|>assistant
{
  "status": "ok",
  "missing_ingredients": [],
  "recipe": {
    "name": "Vegan Tofu Fried Rice",
    "cuisine": "Asian",
    "culinary_preference": "vegan",
    "time": "20m",
    "main_ingredients": ["tofu", "rice", "vegetables"],
    "steps": "1. Press tofu and cut into cubes...\n2. Heat oil in wok...",
    "note": "Add soy sauce to taste"
  },
  "shopping_list": []
}<|im_end|>
```

**Why ChatML format?**
- Matches training paradigm of instruction-tuned models
- Explicit role separation (system/user/assistant)
- Structured JSON output for easy parsing and display
- Better constraint enforcement through system prompts
- Standard format for Hugging Face trainers

### 3. Data Validation and Cleaning

**Validation process**:
1. **Dietary constraint checking**
   - Vegan: No meat, dairy, eggs, honey
   - Vegetarian: No meat, fish
   - Gluten-free: No wheat, flour, barley, rye
   - Dairy-free: No milk, butter, cheese

2. **Violation detection**
   - Pattern matching for forbidden ingredients
   - Context-aware checking (e.g., "almond milk" ≠ violation for vegan)

3. **Data cleaning**
   - Removed ~150 recipes with violations (1.25%)
   - Final dataset: **11,850 clean recipes**
   - 100% dietary compliance verified

**Common violations found**:
- Vegan recipes with honey (45 cases)
- Gluten-free recipes with soy sauce (38 cases)
- Dairy-free recipes with butter (32 cases)
- Vegetarian recipes with chicken broth (35 cases)

### 4. Fine-Tuning on Lambda Labs

**Setup**:
- Platform: **Lambda Labs GPU Cloud**
- Instance: 1x A100 (40GB VRAM)
- Duration: ~3 hours
- Cost: ~$15

**Model Configuration**:
- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Fine-tuning method: **LoRA (Low-Rank Adaptation)**
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05

**Training Hyperparameters**:
- Learning rate: 2e-4
- Batch size: 4 (per device)
- Gradient accumulation: 4 steps
- Epochs: 3
- Optimizer: AdamW (8-bit)
- LR scheduler: Cosine with warmup

**Why LoRA?**
- **Efficient**: Only trains 0.1% of parameters (~35MB adapter vs 6GB full model)
- **Fast**: 3 hours vs days for full fine-tuning
- **Portable**: Easy to share and deploy
- **Memory efficient**: Fits on consumer GPUs

**Training Dataset Split**:
- Train: 9,480 recipes (80%)
- Validation: 1,185 recipes (10%)
- Test: 1,185 recipes (10%)

See [`data_pipeline/04_training/LAMBDA_LABS_SETUP_GUIDE.md`](data_pipeline/04_training/LAMBDA_LABS_SETUP_GUIDE.md) for detailed setup instructions.

### 5. DPO Persona Fine-Tuning

After SFT training, we applied **Direct Preference Optimization (DPO)** to create 6 persona-specific models that align with user preferences for cuisine style and dietary constraints.

**Why DPO?**
- **Preference Learning**: Learns from human preferences instead of ground truth labels
- **Cost-Effective**: Uses preference pairs (chosen vs rejected) generated by LLMs
- **Persona Alignment**: Tailors responses to specific cuisine styles and dietary needs
- **No Reward Model Needed**: Direct optimization unlike RLHF

**6 Personas Trained**:
1. **Korean Spicy** - Emphasizes gochugaru, gochujang, kimchi
2. **Indian Vegetarian** - Plant-based with turmeric, cumin, paneer
3. **Italian Gluten-Free** - Traditional Italian without wheat/gluten
4. **Japanese Low-Sodium** - Umami-rich with reduced salt
5. **Mexican Vegan** - Plant-based with chilies, lime, cilantro
6. **Chinese Keto** - Low-carb with shirataki noodles, cauliflower rice

**Training Process**:
- **Preference Pairs**: ~84 pairs per persona (Groq Llama 3.3 70B)
- **Method**: DPO with LoRA (rank=16, alpha=16)
- **Base**: SFT model from step 4 (llama3b_lambda_lora)
- **Parameters**: Learning rate 5e-5, beta=0.1, 1 epoch
- **Cost**: ~$0.90 total (6 personas)

**Storage**:
- **GCS Bucket**: `gs://pantrypilot-dpo-models/v1.0/`
- **Download Script**: [`download_dpo_models.sh`](download_dpo_models.sh)
- **Size**: 173MB per persona (~1GB total)

See [`data_pipeline/05_dpo_training/README.md`](data_pipeline/05_dpo_training/README.md) for detailed DPO training guide.

### 6. DPO Model Evaluation

**Evaluation Framework**:
- **Evaluator**: Vertex AI (Gemini 2.0 Flash Experimental)
- **Test Cases**: 120 total (20 per persona)
- **Method**: Head-to-head comparison (DPO vs SFT)
- **Location**: us-central1
- **Cost**: ~$0.06 for full evaluation

**Results**:
- **Overall DPO Win Rate**: 75.8% (91/116 completed tests)
- **Production-Ready Models**: 5/6 personas

**Performance by Persona**:

| Persona | Win Rate | Tests | Status |
|---------|----------|-------|--------|
| Korean Spicy | **100%** | 20/20 | ✅ Excellent |
| Mexican Vegan | **85%** | 17/19 | ✅ Strong |
| Japanese Low-Sodium | **80%** | 16/20 | ✅ Strong |
| Indian Vegetarian | **80%** | 16/20 | ✅ Strong |
| Chinese Keto | **80%** | 16/19 | ✅ Strong |
| Italian Gluten-Free | **30%** | 6/18 | ⚠️ Needs Retraining |

**Test Categories**:
- Basic alignment (perfect ingredient matches)
- Constraint stress (banned ingredients present)
- Edge cases (minimal ingredients, ambiguous requests)
- Cross-persona (different cuisine styles)

**Evaluation Reports**:
- HTML Report: [`evaluation/reports/evaluation_report.html`](evaluation/reports/evaluation_report.html)
- Detailed Results: [`evaluation/reports/detailed_results.json`](evaluation/reports/detailed_results.json)
- Summary Stats: [`evaluation/reports/summary_stats.json`](evaluation/reports/summary_stats.json)

See [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) and [`DPO_RESULTS.md`](DPO_RESULTS.md) for comprehensive analysis.

### 7. Model Evaluation (SFT)

**Test Results**:
- Dietary constraint compliance: **100%** (5/5 test cases passed)
- Base model vs Fine-tuned: Significant improvement in constraint adherence
- Response quality: More structured recipes, better ingredient utilization

**Test Categories**:
1. Vegetarian with dairy ✅
2. Vegan (no animal products) ✅
3. Gluten-free ✅
4. Dairy-free ✅
5. Multiple constraints (vegan + gluten-free) ✅

See [`evaluation/reports/lambda_model_test_results.json`](evaluation/reports/lambda_model_test_results.json) for detailed SFT results.

### 8. Deployment: Full-Stack Web Application

**Architecture**:
```
Frontend (React) ←→ Backend (FastAPI) ←→ Model Service (PyTorch + LoRA)
                                      ↓
                                 MongoDB (Inventory + Preferences)
```

**Features**:
- **Recipe Generation**: Natural language requests with inventory
- **Base vs Fine-tuned Comparison**: Side-by-side model outputs
- **Inventory Management**: Track pantry items with quantities
- **Dietary Preferences**: Set and persist dietary constraints
- **Cuisine Preferences**: Request specific cuisine styles

**Technology Stack**:

**Backend**:
- FastAPI (async Python web framework)
- PyTorch (ML inference)
- Transformers + PEFT (model loading)
- MongoDB (async with Motor)

**Frontend**:
- React 18
- Axios (API client)
- CSS modules

**Model Serving**:
- Device: MPS (Apple Silicon) / CUDA / CPU auto-detection
- Model: Llama 3.2 3B Instruct (6GB base + 35MB LoRA)
- Inference: ~2-5 seconds per recipe

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 16+
- Docker (for MongoDB)
- 8GB+ RAM

### 1. Clone Repository

```bash
git clone https://github.com/Shinhunjun/recipe-generation-smallllm.git
cd recipe-generation-smallllm
```

### 2. Download Model and Data Files

**Important**: The trained models and training data are not included in the Git repository due to file size.

#### Download Options:

**🌐 Option 1: Google Cloud Storage** (Recommended for GCP users with `gcloud` CLI)

```bash
# Authenticate with GCP (if not already authenticated)
gcloud auth login

# Download SFT model from GCS
gcloud storage cp -r gs://recipegen-llm-models/llama3b_lambda_lora ./models/

# Download DPO persona models (optional, for advanced usage)
gcloud storage cp -r gs://pantrypilot-dpo-models/v1.0/ ./models/dpo_personas/

# Download training data and pipeline (optional, for reproducibility)
gcloud storage cp -r gs://recipegen-llm-models/data_pipeline/data_pipeline/ ./data_pipeline/

# Verify downloads
ls -lh models/llama3b_lambda_lora/
ls -lh models/dpo_personas/      # If downloaded
ls -lh data_pipeline/data/       # If downloaded
```

**Benefits**: Faster download speeds for GCP users, no manual extraction needed, version-controlled storage.

**📦 Option 2: Google Drive** (For all users)

📦 **SFT Model Files** (52 MB):
- Download: [llama3b_lambda_lora.zip](https://drive.google.com/file/d/1gPWh_ap_OLzseJui477wcmtwjNVOa0XA/view?usp=drive_link)
- Extract to: `models/`

```bash
# After downloading, extract:
unzip llama3b_lambda_lora.zip -d models/
```

📦 **DPO Persona Models** (Optional, ~1GB):
- Use the automated download script:
```bash
chmod +x download_dpo_models.sh
./download_dpo_models.sh models/dpo_personas
```

📊 **Training Data** (45 MB):
- Download: [data.zip](https://drive.google.com/file/d/1S2FmttufCx9OJ5D8LG4G0JRv4in6Fv-1/view?usp=drive_link)
- Extract to: `data_pipeline/data/`

```bash
# After downloading, extract:
unzip data.zip -d data_pipeline/data/
```

**Expected directory structure after extraction**:
```
RecipeGen-LLM/
├── models/
│   ├── llama3b_lambda_lora/           # SFT model (required)
│   │   ├── adapter_model.safetensors
│   │   ├── adapter_config.json
│   │   └── ...
│   └── dpo_personas/                  # DPO models (optional)
│       ├── persona_a_korean_spicy_v1.0/
│       ├── persona_b_indian_veg_v1.0/
│       ├── persona_c_italian_gf_v1.0/
│       ├── persona_d_japanese_lowsodium_v1.0/
│       ├── persona_e_mexican_vegan_v1.0/
│       └── persona_f_chinese_keto_v1.0/
└── data_pipeline/data/
    ├── chat_format/
    │   ├── recipes_train_chat.jsonl
    │   ├── recipes_val_chat.jsonl
    │   └── recipes_test_chat.jsonl
    ├── cleaned/
    │   ├── recipes_train_cleaned.jsonl
    │   ├── recipes_val_cleaned.jsonl
    │   └── recipes_test_cleaned.jsonl
    └── synthetic/
        └── recipes_15k_raw.jsonl
```

### 3. Start MongoDB

```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### 4. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py
```

Server runs on `http://localhost:8000`

### 5. Frontend Setup (New Terminal)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

App opens at `http://localhost:3000`

**For detailed setup instructions, see [SETUP.md](SETUP.md)**

### 6. Usage

1. **Add Inventory**: Go to "Inventory" tab, add ingredients
2. **Set Preferences**: Go to "Settings" tab, set dietary constraints
3. **Generate Recipe**:
   - Enter natural language request (e.g., "Make me a vegan pasta dish")
   - Toggle "Compare models" to see base vs fine-tuned outputs
   - Click "Generate Recipe"

---

## 📋 Recent Updates & Changes

### 🎯 November 2024: DPO Persona Training & Evaluation

Major expansion of the project with persona-specific fine-tuning:

**New Additions**:
- **6 DPO Persona Models**: Cuisine and dietary preference-specific models
  - Korean Spicy, Indian Vegetarian, Italian Gluten-Free
  - Japanese Low-Sodium, Mexican Vegan, Chinese Keto
- **Comprehensive Evaluation System**: Vertex AI-powered automated testing
  - 120 test cases across 6 personas
  - 75.8% overall DPO win rate vs SFT baseline
  - 5/6 models production-ready
- **Complete Documentation**:
  - [DPO_RESULTS.md](DPO_RESULTS.md): Performance analysis
  - [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md): Evaluation system guide
  - [PANTRYPILOT_INTEGRATION.md](PANTRYPILOT_INTEGRATION.md): Integration guide

**Pipeline Expansion**:
- Added `data_pipeline/05_dpo_training/` with full DPO workflow
- Preference pair generation using Groq API (Llama 3.3 70B)
- Automated training scripts for all personas
- Model storage in GCS (`gs://pantrypilot-dpo-models/v1.0/`)

**Cost-Effective Implementation**:
- DPO Training: ~$0.90 (6 personas)
- Vertex AI Evaluation: ~$0.06 (120 tests)
- Total: < $1.00 for complete persona training and evaluation

### 🔄 December 2024: Backend/Frontend Alignment

This project underwent critical updates to align the inference implementation with the training data format:

#### 1. **ChatML Format Migration** (Backend)
**Problem**: Model was trained with ChatML format but backend used Llama 3 format, causing suboptimal performance.

**Changes Made** ([model_service.py](backend/model_service.py)):
- Migrated prompt formatting from Llama 3 to ChatML format (`<|im_start|>`, `<|im_end|>`)
- Simplified inventory input to ingredient names only (matching training data)
- Extract single dietary preference instead of multiple (matching training distribution)
- Updated system prompt to match training exactly

**Before (Llama 3 format)**:
```python
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant...<|eot_id|>
```

**After (ChatML format)**:
```python
<|im_start|>system
You are a recipe generation AI that creates recipes based on user inventory and preferences.<|im_end|>
<|im_start|>user
I have tofu, rice, vegetables. I want a vegan recipe.<|im_end|>
<|im_start|>assistant
```

**Impact**: Model now receives prompts in expected format, significantly improving output quality and dietary constraint adherence.

#### 2. **Structured JSON Output** (Backend)
**Problem**: Backend returned raw text, no structured parsing of model output.

**Changes Made** ([main.py](backend/main.py)):
- Added Pydantic models for type-safe JSON parsing (`RecipeDetails`, `RecipeOutput`, `RecipeResponse`)
- Created `parse_recipe_json()` function with robust error handling
- Supports both perfect JSON and malformed responses (regex extraction)
- Returns structured error messages when parsing fails

**Example JSON Structure**:
```json
{
  "status": "ok",
  "missing_ingredients": [],
  "recipe": {
    "name": "Vegan Tofu Fried Rice",
    "cuisine": "Asian",
    "culinary_preference": "vegan",
    "time": "20m",
    "main_ingredients": ["tofu", "rice", "vegetables"],
    "steps": "Step 1. Press tofu...\nStep 2. Heat oil...",
    "note": "Add soy sauce to taste"
  },
  "shopping_list": []
}
```

**Impact**: Frontend receives structured data, enabling rich UI components instead of plain text display.

#### 3. **Structured Recipe Display** (Frontend)
**Problem**: Frontend displayed raw text in `<pre>` tags, no visual hierarchy.

**Changes Made** ([RecipeGenerator.js](frontend/src/components/RecipeGenerator.js), [RecipeGenerator.css](frontend/src/components/RecipeGenerator.css)):
- Created `RecipeDisplay` component with sections for:
  - Recipe metadata (name, cuisine, time, dietary preference)
  - Main ingredients (grid layout)
  - Step-by-step instructions (numbered boxes)
  - Chef's notes (highlighted)
  - Shopping list (if ingredients missing)
- Added error and warning message displays
- Implemented comparison view for base vs fine-tuned models

**Visual Improvements**:
- Recipe name: Large, bold header
- Metadata: Pills with icons (🍳, ⏱️, 🥗)
- Ingredients: Grid cards with color-coded borders
- Steps: Separate boxes with green accent borders
- Notes: Yellow background for emphasis
- Shopping list: Orange accent borders

**Impact**: Professional, easy-to-read recipe presentation with clear visual hierarchy.

#### 4. **Smart Step Formatting** (Frontend)
**Problem**: Fine-tuned model outputs steps as continuous text ("Step 1. ... Step 2. ...") while base model uses newlines. Both should display consistently.

**Solution** ([RecipeGenerator.js:85-99](frontend/src/components/RecipeGenerator.js#L85-L99)):
```javascript
{(() => {
  // First try to split by newlines
  let steps = recipe.steps.split('\n').filter(s => s.trim());

  // If only one step exists, try to split by "Step N." pattern
  if (steps.length === 1) {
    steps = recipe.steps.split(/Step \d+\./).filter(s => s.trim());
  }

  return steps.map((step, i) => (
    <p key={i} className="step">
      {step.trim().startsWith('Step') ? step.trim() : `Step ${i + 1}. ${step.trim()}`}
    </p>
  ));
})()}
```

**Impact**: Both base and fine-tuned models now display steps in separate, formatted boxes consistently.

---

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

#### 1. Backend: `ModuleNotFoundError`
**Problem**: Python packages not installed.

**Solution**:
```bash
cd backend
source venv/bin/activate  # Important: Activate venv first!
pip install -r requirements.txt
```

#### 2. Docker: `Cannot connect to the Docker daemon`
**Problem**: Docker Desktop not running.

**Solution** (macOS):
```bash
open -a Docker  # Start Docker Desktop
# Wait for Docker to fully start, then:
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

#### 3. Frontend: Port 3000 already in use
**Problem**: Another process is using port 3000.

**Solution**:
```bash
lsof -ti:3000 | xargs kill -9  # Kill existing process
npm start
```

#### 4. Model: All recipes are vegan/vegetarian
**Problem**: Dietary preferences set in Settings persist in MongoDB.

**Check Settings**: Go to Settings tab in the web app and verify dietary restrictions. These are automatically added to every prompt.

**Expected Behavior**: If you set "vegan" in Settings, every recipe will be vegan because the prompt becomes "I have [ingredients]. I want a vegan recipe."

#### 5. Model: Base and fine-tuned give similar recipes
**Explanation**: This is expected behavior when:
- Same inventory constraints
- Same dietary preferences
- Temperature=0.7 (allows limited randomness)
- LoRA modifies only 0.1% of parameters

**Try**:
- Different user requests ("quick breakfast" vs "comfort food dinner")
- Remove dietary restrictions from Settings
- Add more diverse ingredients to inventory

#### 6. Model: Steps not displaying properly
**Solution**: This should be fixed by the smart step formatter. If still broken:
1. Check browser console for errors
2. Verify model output contains either:
   - Newlines: `"Step 1. ...\nStep 2. ..."`
   - OR pattern: `"Step 1. ... Step 2. ..."`

#### 7. Backend: Model loading takes forever
**First Run**: Base model (6GB) downloads from Hugging Face (~5-10 min on fast connection).

**Check Progress**: Look at backend logs for download progress bars.

**Storage**: Ensure you have 10GB+ free disk space (6GB base + 4GB temporary).

#### 8. Frontend: Blank page or React errors
**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

#### 9. Database: Inventory/preferences not saving
**Check MongoDB**:
```bash
docker ps  # Verify mongodb container is running
docker logs mongodb  # Check for errors
```

**Restart MongoDB**:
```bash
docker restart mongodb
```

#### 10. Performance: Recipe generation is slow
**Expected Timing**:
- First run: 5-10 seconds (model loading)
- Subsequent runs: 2-5 seconds per recipe
- Comparison mode: 4-10 seconds (generates twice)

**Device Performance**:
- **MPS (Apple Silicon)**: 2-3 seconds ✅
- **CUDA (NVIDIA GPU)**: 2-4 seconds ✅
- **CPU**: 10-30 seconds ⚠️

**Improve Speed**:
- Use GPU if available (MPS/CUDA detected automatically)
- Disable comparison mode (generates only fine-tuned model)
- Reduce inventory size (fewer items to consider)

---

## 📦 Model Weights

The fine-tuned LoRA adapter is available from multiple sources:

### Download Sources:

**🌐 Google Cloud Storage** (Recommended for GCP environments):
```bash
gcloud storage cp -r gs://recipegen-llm-models/llama3b_lambda_lora ./models/
```
- **Bucket**: `gs://recipegen-llm-models/`
- **Region**: `us-central1`
- **Access**: Public read access (requires `gcloud` authentication)

**📦 Google Drive** (For general users):
- Download: [llama3b_lambda_lora.zip](https://drive.google.com/file/d/1gPWh_ap_OLzseJui477wcmtwjNVOa0XA/view?usp=drive_link)

### Model Contents:

Located in `models/llama3b_lambda_lora/`:
- `adapter_model.safetensors` (35MB) - LoRA adapter weights
- `adapter_config.json` - Adapter configuration
- `tokenizer.json` (16MB) - Tokenizer
- `tokenizer_config.json` (53KB) - Tokenizer configuration
- `special_tokens_map.json` - Special tokens
- `training_args.bin` - Training arguments

**Total size**: ~52 MB

**Base model** (`meta-llama/Llama-3.2-3B-Instruct`, ~6GB) is auto-downloaded from Hugging Face on first run.

---

## 🔬 Evaluation

### SFT Model Evaluation

Run comprehensive dietary constraint tests:

```bash
cd evaluation
python test_lambda_model.py
```

This tests:
- 5 dietary constraint scenarios
- Base vs fine-tuned comparison
- Violation detection
- Recipe quality metrics

Results saved to `evaluation/reports/lambda_model_test_results.json`

### DPO Persona Evaluation

Run automated evaluation using Vertex AI:

```bash
cd evaluation
./run_evaluation.sh YOUR_GCP_PROJECT_ID
```

**What it does**:
- Tests 6 DPO persona models vs SFT baseline
- 120 test cases (20 per persona)
- Head-to-head comparison using Gemini 2.0 Flash
- Generates HTML report with detailed analysis

**Requirements**:
- GCP project with Vertex AI enabled
- Authenticated with `gcloud auth login`
- DPO models downloaded to `models/dpo_personas/`

**Output**:
- `evaluation/reports/evaluation_report.html` - Interactive report
- `evaluation/reports/detailed_results.json` - Full results
- `evaluation/reports/summary_stats.json` - Statistics

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed instructions.

---

## 🌐 Future Deployment (GCP)

Planned architecture for cloud deployment:

```
Cloud Load Balancer
    ↓
Cloud Run (Frontend) ←→ Cloud Run (Backend API)
                              ↓
                         Vertex AI Model Endpoint
                              ↓
                         Cloud Storage (Model)
    ↓
MongoDB Atlas / Cloud SQL
```

**Benefits**:
- Auto-scaling
- Managed infrastructure
- Global CDN
- Serverless pricing

See [`deployment/`](deployment/) for Docker configs.

---

## 📚 Documentation

### Core Pipeline
- **Data Pipeline**: [`data_pipeline/`](data_pipeline/)
  - [01_synthetic_generation](data_pipeline/01_synthetic_generation/README.md) - Recipe data generation
  - [02_chat_conversion](data_pipeline/02_chat_conversion/README.md) - ChatML formatting
  - [03_validation](data_pipeline/03_validation/README.md) - Dietary constraint validation
  - [04_training](data_pipeline/04_training/LAMBDA_LABS_SETUP_GUIDE.md) - SFT training guide
  - [05_dpo_training](data_pipeline/05_dpo_training/README.md) - DPO persona training

### DPO & Evaluation
- **DPO Results**: [DPO_RESULTS.md](DPO_RESULTS.md) - Comprehensive performance analysis
- **Evaluation Guide**: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - How to evaluate DPO models
- **Integration**: [PANTRYPILOT_INTEGRATION.md](PANTRYPILOT_INTEGRATION.md) - PantryPilot integration guide
- **Download Script**: [download_dpo_models.sh](download_dpo_models.sh) - Automated model downloads

### Additional Resources
- **Scenarios**: [SCENARIOS.md](data_pipeline/01_synthetic_generation/SCENARIOS.md) - Recipe generation scenarios
- **Model Architecture**: [model_service.py](backend/model_service.py) - Model service implementation

---

## 🛠️ Technology Choices

### Why Llama 3.2 3B?
- **Size**: Small enough for local inference (6GB), large enough for quality
- **Instruction-tuned**: Pre-trained for chat format
- **Open source**: No API costs, full control
- **Apple Silicon optimized**: Fast inference on MPS

### Why LoRA?
- **Parameter-efficient**: 35MB adapter vs 6GB full model
- **Fast training**: 3 hours vs days
- **Easy deployment**: Swap adapters without reloading base model
- **Research-proven**: PEFT library by Hugging Face

### Why Groq API?
- **Speed**: 10-100x faster than OpenAI for synthetic data generation
- **Cost**: Lower pricing for high-volume API calls
- **Quality**: Llama 3.1 8B comparable to GPT-3.5

### Why FastAPI?
- **Async**: Non-blocking I/O for model inference
- **Type safety**: Pydantic models
- **OpenAPI**: Auto-generated docs
- **Performance**: Faster than Flask/Django

### Why React?
- **Component-based**: Reusable UI components
- **Ecosystem**: Rich libraries (Axios, React Router)
- **Developer experience**: Hot reload, debugging tools

---

## 📝 License

MIT License

---

## 👥 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 🙏 Acknowledgments

- **Groq** for fast and affordable LLM API
- **Lambda Labs** for GPU cloud infrastructure
- **Hugging Face** for Transformers and PEFT libraries
- **Meta** for open-sourcing Llama 3 models

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Llama 3, PyTorch, FastAPI, and React**
