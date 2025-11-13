"""
PyTorch Model Service with Llama 3B + LoRA adapter
Supports base model vs fine-tuned model comparison
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from typing import List, Dict, Optional


class ModelService:
    def __init__(self, base_model_id: str, adapter_path: str):
        """
        Initialize model service with Llama 3B base model and LoRA adapter

        Args:
            base_model_id: HuggingFace model ID (meta-llama/Llama-3.2-3B-Instruct)
            adapter_path: Path to LoRA adapter weights
        """
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🖥️  Using device: {self.device}")
        print(f"📥 Loading base model: {base_model_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        print(f"✅ Base model loaded (~6GB memory)")

        # Load fine-tuned model with LoRA adapter
        if Path(adapter_path).exists():
            print(f"📥 Loading LoRA adapter: {adapter_path}")
            self.finetuned_model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.finetuned_model.eval()
            print(f"✅ Fine-tuned model loaded (base + 35MB adapter)")
        else:
            print(f"⚠️  LoRA adapter not found at {adapter_path}")
            self.finetuned_model = None

    def _format_prompt(self, inventory: List[Dict], preferences: Dict, user_request: str) -> str:
        """Format user input into Llama 3 chat template"""

        # Build inventory list
        inventory_items = []
        for item in inventory:
            item_str = item['name']
            if item.get('quantity') and item.get('unit'):
                item_str += f" ({item['quantity']} {item['unit']})"
            inventory_items.append(item_str)

        inventory_str = ", ".join(inventory_items)

        # Extract dietary restrictions
        dietary_restrictions = []
        if preferences.get("dietary_restrictions"):
            dietary_restrictions = preferences["dietary_restrictions"]

        # Format dietary constraints
        if dietary_restrictions:
            constraints_str = ", ".join(dietary_restrictions)
        else:
            constraints_str = "None"

        # System instruction with dietary constraint enforcement
        system_prompt = "You are a helpful recipe assistant. Generate recipes based on available ingredients and dietary constraints."

        # Add explicit constraint warnings for critical diets
        constraint_warnings = ""
        if "vegan" in [d.lower() for d in dietary_restrictions]:
            constraint_warnings += "\n\nIMPORTANT: This recipe must be VEGAN. Do not use any animal products including:\n- Meat, poultry, fish, seafood\n- Dairy (milk, cheese, butter, cream, yogurt)\n- Eggs\n- Honey"
        elif "vegetarian" in [d.lower() for d in dietary_restrictions]:
            constraint_warnings += "\n\nIMPORTANT: This recipe must be VEGETARIAN. Do not use:\n- Meat, poultry, fish, seafood\n- Gelatin or any meat-based products"

        if "gluten-free" in [d.lower() for d in dietary_restrictions]:
            constraint_warnings += "\n\nIMPORTANT: This recipe must be GLUTEN-FREE. Do not use:\n- Wheat, flour, bread, pasta\n- Barley, rye\n- Soy sauce (use gluten-free alternatives)"

        if "dairy-free" in [d.lower() for d in dietary_restrictions]:
            constraint_warnings += "\n\nIMPORTANT: This recipe must be DAIRY-FREE. Do not use:\n- Milk, butter, cream, cheese, yogurt\n- Any dairy-based products"

        # User request
        if user_request:
            user_content = f"{user_request}\n\nAvailable ingredients: {inventory_str}\nDietary constraints: {constraints_str}"
        else:
            user_content = f"Generate a recipe using these ingredients: {inventory_str}\nDietary constraints: {constraints_str}"

        # Llama 3 chat template
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}{constraint_warnings}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        return prompt

    def generate_recipe(
        self,
        inventory: List[Dict],
        preferences: Dict,
        user_request: str = "",
        use_finetuned: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a recipe using either base or fine-tuned model

        Args:
            inventory: List of available ingredients
            preferences: User preferences including dietary restrictions
            user_request: Optional natural language request
            use_finetuned: Use fine-tuned model (True) or base model (False)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated recipe text
        """
        prompt = self._format_prompt(inventory, preferences, user_request)

        # Select model
        if use_finetuned and self.finetuned_model:
            model = self.finetuned_model
            model_name = "Fine-tuned"
        else:
            model = self.base_model
            model_name = "Base"

        print(f"🤖 Generating recipe with {model_name} model...")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract only assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in full_output:
            response = full_output.split("<|start_header_id|>assistant<|end_header_id|>")[1]
            response = response.split("<|eot_id|>")[0].strip()
            return response

        return full_output

    def generate_comparison(
        self,
        inventory: List[Dict],
        preferences: Dict,
        user_request: str = "",
        max_tokens: int = 512
    ) -> Dict[str, str]:
        """
        Generate recipes with both base and fine-tuned models for comparison

        Returns:
            Dict with "base" and "finetuned" keys containing respective recipes
        """
        base_recipe = self.generate_recipe(
            inventory, preferences, user_request,
            use_finetuned=False, max_tokens=max_tokens
        )

        finetuned_recipe = self.generate_recipe(
            inventory, preferences, user_request,
            use_finetuned=True, max_tokens=max_tokens
        )

        return {
            "base": base_recipe,
            "finetuned": finetuned_recipe
        }

    def cleanup(self):
        """Free up GPU memory"""
        del self.base_model
        if self.finetuned_model:
            del self.finetuned_model
        torch.cuda.empty_cache()
        print("✅ Models unloaded, memory freed")


# Global model instance (lazy loading)
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create global model service instance"""
    global _model_service

    if _model_service is None:
        BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
        ADAPTER_PATH = "../models/llama3b_lambda_lora"

        _model_service = ModelService(BASE_MODEL, ADAPTER_PATH)

    return _model_service
