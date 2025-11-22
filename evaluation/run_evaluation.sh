#!/bin/bash
# Quick start script for DPO evaluation
# Usage: ./run_evaluation.sh YOUR_GCP_PROJECT_ID

set -e  # Exit on error

PROJECT_ID=$1

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: GCP project ID required"
    echo "Usage: ./run_evaluation.sh YOUR_GCP_PROJECT_ID"
    echo ""
    echo "Example:"
    echo "  ./run_evaluation.sh my-mlops-project"
    exit 1
fi

echo "=================================="
echo "DPO Persona Evaluation Runner"
echo "=================================="
echo ""
echo "GCP Project: $PROJECT_ID"
echo ""

# Check if authenticated
echo "📝 Checking GCP authentication..."
if ! gcloud auth application-default print-access-token &> /dev/null; then
    echo "⚠️  Not authenticated. Running authentication..."
    gcloud auth application-default login
fi

echo "✅ Authenticated"
echo ""

# Enable Vertex AI API
echo "🔧 Ensuring Vertex AI API is enabled..."
gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID
echo "✅ Vertex AI enabled"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r evaluation/requirements.txt

echo "✅ Dependencies installed"
echo ""

# Run evaluation
echo "=================================="
echo "🚀 Starting Evaluation"
echo "=================================="
echo ""
echo "Configuration:"
echo "  - Personas: All (6 personas)"
echo "  - Tests per persona: 20"
echo "  - Evaluators: Gemini Flash, Claude Haiku, Claude Sonnet"
echo "  - Total tests: 120"
echo "  - Estimated cost: ~$3-4"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

echo ""
echo "🏃 Running evaluation (this may take 30-60 minutes)..."
echo ""

python evaluation/evaluate_dpo_personas.py \
    --project_id $PROJECT_ID \
    --personas all \
    --count 20 \
    --evaluators all \
    --output_dir evaluation/reports

echo ""
echo "=================================="
echo "✅ Evaluation Complete!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - evaluation/reports/evaluation_report.html (open in browser)"
echo "  - evaluation/reports/detailed_results.json"
echo "  - evaluation/reports/summary_stats.json"
echo ""
echo "Open the HTML report:"
echo "  open evaluation/reports/evaluation_report.html"
echo ""
