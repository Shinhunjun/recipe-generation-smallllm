"""
PantryPilot Recipe Generator API
FastAPI backend with MLX model inference
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
from pathlib import Path

from model_service import ModelService
from database import Database

app = FastAPI(title="PantryPilot Recipe Generator")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_service: Optional[ModelService] = None
db: Optional[Database] = None

# Request/Response models
class GenerateRecipeRequest(BaseModel):
    user_request: str  # Natural language request (e.g., "I want something healthy for dinner")
    compare: bool = False

class RecipeResponse(BaseModel):
    recipe: str
    base_recipe: Optional[str] = None  # Only if compare=True

class InventoryItem(BaseModel):
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None

class UserPreferences(BaseModel):
    dietary_restrictions: List[str] = []
    cooking_style: str = "balanced"
    custom_preferences: str = ""

@app.on_event("startup")
async def startup_event():
    """Initialize model and database on server start"""
    global model_service, db

    print("🚀 Initializing PantryPilot API...")

    # Load model - V3 checkpoint 5500 (66.7% dietary constraint accuracy, best working checkpoint)
    model_path = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    adapter_path = Path(__file__).parent.parent / "models" / "phi3-recipe-lora-v3"

    print(f"📍 Adapter path: {adapter_path}")
    print(f"📍 Using V3 Checkpoint 5500 (66.7% dietary constraint accuracy - best working checkpoint)")
    model_service = ModelService(model_path, str(adapter_path))
    print("✅ Model loaded")

    # Connect to MongoDB
    db = Database()
    await db.connect()
    print("✅ Database connected")

    # Initialize demo inventory
    await db.init_demo_inventory()
    print("✅ Demo inventory initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    if db:
        await db.disconnect()
    print("👋 Server shutdown")

@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "service": "PantryPilot Recipe Generator"}

@app.post("/api/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: GenerateRecipeRequest):
    """Generate recipe based on user's natural language request and inventory"""
    try:
        # Get current inventory and preferences
        inventory = await db.get_inventory()
        preferences = await db.get_preferences()

        if request.compare:
            # Generate with both base and fine-tuned models
            base_output, finetuned_output = model_service.generate_comparison(
                inventory,
                preferences,
                request.user_request
            )
            return RecipeResponse(
                recipe=finetuned_output,
                base_recipe=base_output
            )
        else:
            # Generate with fine-tuned model only
            output = model_service.generate_recipe(
                inventory,
                preferences,
                request.user_request,
                use_finetuned=True
            )
            return RecipeResponse(recipe=output)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/inventory")
async def get_inventory():
    """Get current inventory"""
    inventory = await db.get_inventory()
    return {"inventory": inventory}

@app.post("/api/inventory")
async def add_inventory_item(item: InventoryItem):
    """Add item to inventory"""
    result = await db.add_inventory_item(item.dict())
    return {"success": True, "item": result}

@app.delete("/api/inventory/{item_name}")
async def remove_inventory_item(item_name: str):
    """Remove item from inventory"""
    await db.remove_inventory_item(item_name)
    return {"success": True, "removed": item_name}

@app.get("/api/preferences")
async def get_preferences():
    """Get user preferences"""
    prefs = await db.get_preferences()
    return prefs

@app.post("/api/preferences")
async def update_preferences(preferences: UserPreferences):
    """Update user preferences"""
    result = await db.update_preferences(preferences.dict())
    return {"success": True, "preferences": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
