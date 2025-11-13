# Deployment Guide

This directory contains Docker configurations for deploying RecipeGen-LLM locally or to the cloud.

## Local Deployment with Docker Compose

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM (for model inference)
- (Optional) NVIDIA GPU with CUDA support

### Quick Start

```bash
cd deployment

# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- MongoDB: localhost:27017

### Configuration

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - MONGODB_URI=mongodb://mongodb:27017
  - MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
  - ADAPTER_PATH=/app/models/llama3b_lambda_lora
```

### GPU Support

Uncomment GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requires NVIDIA Docker runtime:
```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## Cloud Deployment (GCP)

### Architecture

```
Cloud Load Balancer
    ↓
Cloud Run (Frontend) ←→ Cloud Run (Backend)
                            ↓
                      Vertex AI Endpoint
                            ↓
                      Cloud Storage (Model)
    ↓
MongoDB Atlas
```

### Step 1: Prepare Model

Upload LoRA adapter to Cloud Storage:

```bash
gsutil mb gs://recipe-gen-models
gsutil -m cp -r models/llama3b_lambda_lora gs://recipe-gen-models/
```

### Step 2: Deploy Backend to Cloud Run

```bash
# Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/recipe-backend ../backend

# Deploy to Cloud Run
gcloud run deploy recipe-backend \
  --image gcr.io/PROJECT_ID/recipe-backend \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300 \
  --set-env-vars MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/recipe \
  --set-env-vars MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  --set-env-vars ADAPTER_PATH=/app/models/llama3b_lambda_lora \
  --allow-unauthenticated
```

### Step 3: Deploy Frontend to Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/recipe-frontend ../frontend

# Deploy
gcloud run deploy recipe-frontend \
  --image gcr.io/PROJECT_ID/recipe-frontend \
  --platform managed \
  --region us-central1 \
  --set-env-vars REACT_APP_API_URL=https://recipe-backend-xxx.run.app \
  --allow-unauthenticated
```

### Step 4: MongoDB Atlas

1. Create cluster at https://cloud.mongodb.com
2. Whitelist Cloud Run IP ranges
3. Create database user
4. Get connection string
5. Update backend env vars

### Alternative: Vertex AI Model Endpoint

For production-grade model serving:

```bash
# Upload model to Vertex AI Model Registry
gcloud ai models upload \
  --region=us-central1 \
  --display-name=llama3b-recipe-lora \
  --container-image-uri=gcr.io/PROJECT_ID/recipe-model-server \
  --artifact-uri=gs://recipe-gen-models/llama3b_lambda_lora

# Deploy endpoint
gcloud ai endpoints create \
  --region=us-central1 \
  --display-name=recipe-endpoint

gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID \
  --display-name=llama3b-deployment \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1
```

---

## Environment Variables

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MODEL_PATH` | `meta-llama/Llama-3.2-3B-Instruct` | Base model ID |
| `ADAPTER_PATH` | `../models/llama3b_lambda_lora` | LoRA adapter path |
| `PORT` | `8000` | API server port |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `REACT_APP_API_URL` | `http://localhost:8000` | Backend API URL |
| `PORT` | `3000` | Development server port |

---

## Scaling Considerations

### Horizontal Scaling

**Backend**:
- Use load balancer (Cloud Run auto-scales)
- Stateless design (no session storage)
- Share MongoDB connection pool

**Model Inference**:
- Cache model in memory (don't reload per request)
- Use GPU for faster inference
- Consider batch inference for multiple requests

### Vertical Scaling

**Memory Requirements**:
- Base model: ~6GB
- LoRA adapter: ~35MB
- MongoDB: ~500MB
- Total: ~7-8GB RAM minimum

**CPU/GPU**:
- CPU inference: 2-5s per recipe
- MPS (Apple Silicon): 1-3s per recipe
- CUDA (NVIDIA GPU): 0.5-1s per recipe

### Cost Optimization

**Cloud Run**:
- Use minimum instances = 0 (pay only for requests)
- Set max instances based on traffic
- Use Vertex AI for high-traffic scenarios

**Storage**:
- Use Cloud Storage for model artifacts
- Enable lifecycle policies for logs
- Use MongoDB Atlas free tier for dev/testing

---

## Monitoring

### Health Checks

**Backend**:
```bash
curl http://localhost:8000/health
```

**Frontend**:
```bash
curl http://localhost:3000
```

### Logging

**Docker Compose**:
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```

**Cloud Run**:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=recipe-backend" --limit 50
```

### Metrics

Track in production:
- Request latency (p50, p95, p99)
- Model inference time
- Memory usage
- Error rates
- User sessions

---

## Troubleshooting

### Model Out of Memory

**Solution**: Reduce batch size or use quantization

```python
# 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)
```

### Slow Inference

**Solution**: Use GPU or optimize generation params

```python
# Faster generation
outputs = model.generate(
    **inputs,
    max_new_tokens=256,  # Reduce from 512
    do_sample=False,     # Greedy decoding
    num_beams=1          # No beam search
)
```

### MongoDB Connection Timeout

**Solution**: Check network and increase timeout

```python
client = motor.motor_asyncio.AsyncIOMotorClient(
    uri,
    serverSelectionTimeoutMS=5000
)
```

---

## Security

### Production Checklist

- [ ] Use HTTPS (SSL/TLS certificates)
- [ ] Enable CORS only for trusted origins
- [ ] Use environment variables for secrets (no hardcoding)
- [ ] Implement rate limiting
- [ ] Add authentication (OAuth, JWT)
- [ ] Use MongoDB authentication
- [ ] Enable Cloud Run IAM authentication
- [ ] Scan Docker images for vulnerabilities
- [ ] Regular dependency updates
- [ ] Monitor for suspicious activity

### Example: Rate Limiting

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/api/generate-recipe")
@limiter.limit("10/minute")  # 10 requests per minute
async def generate_recipe(request: RecipeRequest):
    ...
```

---

## Performance Tuning

### Backend Optimization

1. **Model caching**: Load model once, reuse for all requests
2. **Async I/O**: Use async database calls
3. **Connection pooling**: Reuse MongoDB connections
4. **Prompt caching**: Cache common prompts
5. **Response streaming**: Stream long responses

### Frontend Optimization

1. **Code splitting**: Lazy load components
2. **Asset optimization**: Compress images, minify JS/CSS
3. **CDN**: Serve static assets from CDN
4. **Service worker**: Cache API responses
5. **Debouncing**: Debounce user input

---

## Backup and Recovery

### MongoDB Backups

**Automated backups**:
```bash
# Daily backup cron job
0 2 * * * docker exec recipe-mongodb mongodump --out /backup/$(date +\%Y\%m\%d)
```

**Restore**:
```bash
docker exec recipe-mongodb mongorestore /backup/20241113
```

### Model Checkpoints

Store model checkpoints in version-controlled storage:
```bash
gsutil -m cp -r models/ gs://recipe-gen-backups/models-$(date +%Y%m%d)/
```

---

For questions or issues, see main [README.md](../README.md) or open a GitHub issue.
