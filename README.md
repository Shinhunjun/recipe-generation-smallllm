# RecipeGen-LLM: Fine-Tuned Recipe Generation with Llama 3B

A complete end-to-end pipeline for generating personalized recipes using **Llama 3.2 3B Instruct** fine-tuned with **LoRA** on synthetic recipe data.

## 🎯 Project Overview

This project demonstrates:
- **Synthetic data generation** using Groq API (Llama 3.1 8B)
- **Dietary constraint validation** and data cleaning
- **LoRA fine-tuning** on Lambda Labs GPU cluster
- **Full-stack web application** with base vs fine-tuned model comparison
- **Deployment-ready** architecture (local + future GCP)

## 📊 Key Results

- **12,000 synthetic recipes** generated across 6 diverse scenarios
- **100% dietary constraint compliance** after validation and cleaning
- **Fine-tuned model** trained on Lambda Labs (Llama 3.2 3B + LoRA)
- **Web app** with real-time recipe generation and inventory management

---

## 🏗️ Project Structure

```
RecipeGen-LLM/
├── data_pipeline/                 # Data generation and training pipeline
│   ├── 01_synthetic_generation/   # Groq API recipe generation
│   ├── 02_chat_conversion/        # Convert to Llama 3 chat format
│   ├── 03_validation/             # Dietary constraint validation
│   └── 04_training/               # Lambda Labs fine-tuning
├── models/
│   └── llama3b_lambda_lora/       # Trained LoRA adapter (35MB)
├── backend/                       # FastAPI + PyTorch model service
├── frontend/                      # React web application
├── evaluation/                    # Model testing and comparison
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

Converts structured JSON to **Llama 3 chat template**:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful recipe assistant. Generate recipes based on available ingredients and dietary constraints.

IMPORTANT: This recipe must be VEGAN. Do not use any animal products...<|eot_id|><|start_header_id|>user<|end_header_id|>

Generate a recipe using these ingredients: tofu, rice, vegetables
Dietary constraints: vegan<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Recipe Name: Vegan Tofu Fried Rice
Ingredients:
- tofu: 200g
...<|eot_id|>
```

**Why chat format?**
- Matches training paradigm of instruction-tuned models
- Explicit role separation (system/user/assistant)
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

### 5. Model Evaluation

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

See [`evaluation/reports/lambda_model_test_results.json`](evaluation/reports/lambda_model_test_results.json) for detailed results.

### 6. Deployment: Full-Stack Web Application

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

### 2. Start MongoDB

```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### 3. Backend Setup

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

### 4. Frontend Setup (New Terminal)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

App opens at `http://localhost:3000`

**For detailed setup instructions, see [SETUP.md](SETUP.md)**

### 5. Usage

1. **Add Inventory**: Go to "Inventory" tab, add ingredients
2. **Set Preferences**: Go to "Settings" tab, set dietary constraints
3. **Generate Recipe**:
   - Enter natural language request (e.g., "Make me a vegan pasta dish")
   - Toggle "Compare models" to see base vs fine-tuned outputs
   - Click "Generate Recipe"

---

## 📦 Model Weights

The fine-tuned LoRA adapter is included in `models/llama3b_lambda_lora/`:
- `adapter_model.safetensors` (35MB)
- `adapter_config.json`
- Tokenizer files

**Base model** (`meta-llama/Llama-3.2-3B-Instruct`) is auto-downloaded from Hugging Face on first run.

---

## 🔬 Evaluation

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

- **Data Pipeline**: [`data_pipeline/`](data_pipeline/)
  - [01_synthetic_generation](data_pipeline/01_synthetic_generation/README.md)
  - [02_chat_conversion](data_pipeline/02_chat_conversion/README.md)
  - [03_validation](data_pipeline/03_validation/README.md)
  - [04_training](data_pipeline/04_training/LAMBDA_LABS_SETUP_GUIDE.md)

- **Scenarios**: [SCENARIOS.md](data_pipeline/01_synthetic_generation/SCENARIOS.md)

- **Model Architecture**: [model_service.py](backend/model_service.py)

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
