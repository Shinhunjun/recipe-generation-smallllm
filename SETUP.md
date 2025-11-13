# Setup Guide

Complete setup instructions for RecipeGen-LLM local development.

## Prerequisites

- Python 3.10 or higher
- Node.js 16 or higher
- Docker (for MongoDB)
- 8GB+ RAM (for model inference)

## Step 1: Clone Repository

```bash
git clone https://github.com/Shinhunjun/recipe-generation-smallllm.git
cd recipe-generation-smallllm
```

## Step 2: Backend Setup

### Create Virtual Environment

```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Expected installation time**: 5-10 minutes (depending on internet speed)

**Package sizes**:
- torch: ~2GB
- transformers: ~500MB
- Other packages: ~200MB
- **Total**: ~2.7GB

### Download Model (First Run)

The base model (`meta-llama/Llama-3.2-3B-Instruct`) will be downloaded automatically on first run (~6GB).

To pre-download:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

**Note**: You may need to accept the Llama 3 license on Hugging Face first.

## Step 3: Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install
```

**Expected installation time**: 2-3 minutes

## Step 4: Database Setup

### Option A: Docker (Recommended)

```bash
# Start MongoDB container
docker run -d -p 27017:27017 --name recipe-mongodb mongo:latest

# Verify it's running
docker ps
```

### Option B: Local MongoDB

If you have MongoDB installed locally:
```bash
# Start MongoDB service
# macOS:
brew services start mongodb-community

# Linux:
sudo systemctl start mongod

# Windows:
net start MongoDB
```

### Verify Connection

```bash
# Using MongoDB Shell (if installed)
mongosh

# Or using Docker
docker exec -it recipe-mongodb mongosh
```

## Step 5: Environment Configuration (Optional)

### Backend Environment Variables

Create `backend/.env` file:

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017

# Model paths
MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
ADAPTER_PATH=../models/llama3b_lambda_lora

# Server
PORT=8000
HOST=0.0.0.0
```

### Frontend Environment Variables

Create `frontend/.env` file:

```env
REACT_APP_API_URL=http://localhost:8000
```

## Step 6: Run Application

### Start Backend

```bash
cd backend

# Activate virtual environment (if not already activated)
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Run server
python main.py
```

**Expected output**:
```
🖥️  Using device: mps
📥 Loading base model: meta-llama/Llama-3.2-3B-Instruct
✅ Base model loaded (~6GB memory)
📥 Loading LoRA adapter: ../models/llama3b_lambda_lora
✅ Fine-tuned model loaded (base + 35MB adapter)
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Start Frontend (New Terminal)

```bash
cd frontend

# Start development server
npm start
```

**Expected output**:
```
Compiled successfully!

You can now view recipe-app in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

## Step 7: Verify Installation

### Backend Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy"}
```

### Frontend

Open browser to: http://localhost:3000

You should see the Recipe Generator app with 3 tabs:
- Generate Recipe
- Inventory
- Settings

## Troubleshooting

### Backend Issues

#### "ModuleNotFoundError: No module named 'torch'"

**Solution**: Activate virtual environment
```bash
source backend/venv/bin/activate  # macOS/Linux
```

#### "Out of memory" when loading model

**Solution**: Your system doesn't have enough RAM (need 8GB+). Try:
1. Close other applications
2. Use 4-bit quantization (edit `model_service.py`):
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)
```

#### "LoRA adapter not found"

**Solution**: The model files are not in the repository (too large for Git). You need to either:
1. Train your own model following `data_pipeline/04_training/LAMBDA_LABS_SETUP_GUIDE.md`
2. Download pre-trained adapter (if available separately)
3. Use base model only (comment out adapter loading in `model_service.py`)

#### "Cannot connect to MongoDB"

**Solution**:
```bash
# Check if MongoDB is running
docker ps

# If not, start it
docker run -d -p 27017:27017 --name recipe-mongodb mongo:latest
```

### Frontend Issues

#### "npm install" fails

**Solution**: Clear npm cache
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

#### "Module not found" errors

**Solution**: Reinstall dependencies
```bash
cd frontend
rm -rf node_modules
npm install
```

#### Cannot connect to backend API

**Solution**: Check backend is running
```bash
curl http://localhost:8000/health
```

If not running, start backend first.

## Directory Structure After Setup

```
recipe-generation-smallllm/
├── backend/
│   ├── venv/                    # Virtual environment (created)
│   ├── main.py
│   ├── model_service.py
│   ├── database.py
│   └── requirements.txt
├── frontend/
│   ├── node_modules/            # Node packages (created)
│   ├── src/
│   ├── public/
│   └── package.json
├── models/
│   └── llama3b_lambda_lora/     # LoRA adapter (35MB)
├── data_pipeline/
└── deployment/
```

## Virtual Environment Best Practices

### Activating/Deactivating

```bash
# Activate
source backend/venv/bin/activate  # macOS/Linux
backend\venv\Scripts\activate     # Windows

# Deactivate (when done)
deactivate
```

### Checking Active Environment

```bash
# Should show path to venv
which python

# Should be Python 3.10+
python --version
```

### Adding New Dependencies

```bash
# Install new package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

### Recreating Environment

If environment gets corrupted:
```bash
# Remove old environment
rm -rf backend/venv

# Create new one
python3 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt
```

## Development Workflow

### Daily Development

```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
python main.py

# Terminal 2: Frontend
cd frontend
npm start

# Terminal 3: MongoDB (if needed)
docker start recipe-mongodb
```

### Stopping Services

```bash
# Stop backend: Ctrl+C in backend terminal

# Stop frontend: Ctrl+C in frontend terminal

# Stop MongoDB
docker stop recipe-mongodb
```

### Updating Code

```bash
# Pull latest changes
git pull origin main

# Update backend dependencies (if requirements.txt changed)
cd backend
source venv/bin/activate
pip install -r requirements.txt

# Update frontend dependencies (if package.json changed)
cd frontend
npm install
```

## Next Steps

- Read [README.md](README.md) for project overview
- Explore [data_pipeline/](data_pipeline/) for data generation pipeline
- Check [deployment/](deployment/) for Docker deployment
- Review [evaluation/](evaluation/) for model testing

## Getting Help

- Check [Troubleshooting](#troubleshooting) section above
- Review main [README.md](README.md)
- Open issue on GitHub
