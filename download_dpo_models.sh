#!/bin/bash
# Download DPO Models from GCS
#
# This script downloads the 6 DPO persona models from Google Cloud Storage.
# Models are stored at: gs://pantrypilot-dpo-models/v1.0/
#
# Usage:
#   ./download_dpo_models.sh [output_directory]
#
# Default output: ./models/dpo_personas/

set -e

# Configuration
GCS_BUCKET="gs://pantrypilot-dpo-models/v1.0"
DEFAULT_OUTPUT_DIR="./models/dpo_personas"
OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  DPO Persona Models Download${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo -e "${YELLOW}⚠️  gsutil not found. Please install Google Cloud SDK:${NC}"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check GCP authentication
if ! gsutil ls ${GCS_BUCKET} &> /dev/null; then
    echo -e "${YELLOW}⚠️  Authentication required. Running:${NC}"
    echo "   gcloud auth application-default login"
    gcloud auth application-default login
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}✅ Output directory: ${OUTPUT_DIR}${NC}"
echo

# Persona models to download
PERSONAS=(
    "persona_a_korean_spicy_v1.0"
    "persona_b_indian_veg_v1.0"
    "persona_c_italian_gf_v1.0"
    "persona_d_japanese_lowsodium_v1.0"
    "persona_e_mexican_vegan_v1.0"
    "persona_f_chinese_keto_v1.0"
)

# Download each persona model
for persona in "${PERSONAS[@]}"; do
    echo -e "${BLUE}📥 Downloading ${persona}...${NC}"

    # Check if already exists
    if [ -d "${OUTPUT_DIR}/${persona}" ]; then
        echo -e "${YELLOW}   ⚠️  ${persona} already exists. Skipping...${NC}"
        continue
    fi

    # Download from GCS
    gsutil -m cp -r "${GCS_BUCKET}/${persona}" "${OUTPUT_DIR}/"

    echo -e "${GREEN}   ✅ Downloaded ${persona}${NC}"
    echo
done

# Print summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Download Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo "Models downloaded to: ${OUTPUT_DIR}"
echo
echo "Available models:"
ls -1 "${OUTPUT_DIR}"
echo
echo "Total size:"
du -sh "${OUTPUT_DIR}"
echo

# Print usage instructions
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Next Steps${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo "To use these models:"
echo "1. Load base model: unsloth/Llama-3.2-3B-Instruct"
echo "2. Load persona adapter:"
echo "   from peft import PeftModel"
echo "   model = PeftModel.from_pretrained(base_model, '${OUTPUT_DIR}/persona_a_korean_spicy_v1.0')"
echo
echo "See EVALUATION_GUIDE.md for evaluation results and usage examples."
echo
