import React, { useState } from 'react';
import axios from 'axios';
import './RecipeGenerator.css';

const API_BASE = 'http://localhost:8000';

function RecipeGenerator({ inventory, preferences }) {
  const [userRequest, setUserRequest] = useState('');
  const [compareMode, setCompareMode] = useState(true); // Default to comparison
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleGenerate = async () => {
    if (!userRequest.trim()) {
      alert('Please enter what you\'d like to cook');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE}/api/generate-recipe`, {
        user_request: userRequest,
        compare: compareMode
      });

      setResult(response.data);
    } catch (error) {
      console.error('Failed to generate recipe:', error);
      alert('Failed to generate recipe. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleGenerate();
    }
  };

  const RecipeDisplay = ({ recipeData, title }) => {
    const { status, missing_ingredients, recipe, shopping_list } = recipeData;

    return (
      <div className="card result-card">
        <h3>{title}</h3>

        {status === "error" && (
          <div className="error-message">
            <strong>⚠️ Error:</strong> Failed to generate recipe properly
          </div>
        )}

        {status === "no_match" && missing_ingredients.length > 0 && (
          <div className="warning-message">
            <strong>⚠️ Missing ingredients:</strong> {missing_ingredients.join(", ")}
          </div>
        )}

        <div className="recipe-display">
          <div className="recipe-header">
            <h4 className="recipe-name">{recipe.name}</h4>
            <div className="recipe-meta">
              <span className="meta-item">🍳 {recipe.cuisine}</span>
              <span className="meta-item">⏱️ {recipe.time}</span>
              <span className="meta-item">🥗 {recipe.culinary_preference}</span>
            </div>
          </div>

          <div className="recipe-section">
            <h5>Main Ingredients:</h5>
            <ul className="ingredients-list">
              {recipe.main_ingredients.map((ing, i) => (
                <li key={i}>{ing}</li>
              ))}
            </ul>
          </div>

          <div className="recipe-section">
            <h5>Instructions:</h5>
            <div className="recipe-steps">
              {recipe.steps.split('\n').map((step, i) => (
                step.trim() && <p key={i} className="step">{step}</p>
              ))}
            </div>
          </div>

          {recipe.note && (
            <div className="recipe-section note">
              <strong>💡 Note:</strong> {recipe.note}
            </div>
          )}

          {shopping_list && shopping_list.length > 0 && (
            <div className="recipe-section">
              <h5>🛒 Shopping List:</h5>
              <ul className="shopping-list">
                {shopping_list.map((item, i) => (
                  <li key={i}>
                    {item.name} {item.quantity && `(${item.quantity})`}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="recipe-generator">
      <div className="card">
        <h2>🎯 Generate Recipe</h2>
        <p className="description">
          Tell me what you'd like to cook, and I'll create a recipe using ingredients from your pantry.
        </p>

        <div className="input-section">
          <label htmlFor="user-request">What would you like to cook?</label>
          <textarea
            id="user-request"
            placeholder="e.g., 'I want something healthy for dinner' or 'Quick breakfast ideas' or 'Comfort food for tonight'"
            value={userRequest}
            onChange={(e) => setUserRequest(e.target.value)}
            onKeyPress={handleKeyPress}
            rows={4}
          />
          <p className="hint">
            The system will automatically reference your inventory ({inventory.length} items) and preferences to create a recipe.
          </p>
        </div>

        <div className="compare-toggle">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={compareMode}
              onChange={(e) => setCompareMode(e.target.checked)}
            />
            <span>Compare Base vs Fine-tuned Model</span>
          </label>
          <p className="hint">
            {compareMode
              ? 'Will show outputs from both the base Llama 3.2 3B model and Lambda-trained LoRA model side-by-side'
              : 'Will only show output from the Lambda-trained LoRA model'}
          </p>
        </div>

        <button
          className="primary generate-button"
          onClick={handleGenerate}
          disabled={loading || !userRequest.trim()}
        >
          {loading ? 'Generating...' : '✨ Generate Recipe'}
        </button>
      </div>

      {loading && (
        <div className="card loading-card">
          <div className="spinner"></div>
          <p className="loading">
            {compareMode ? 'Generating recipes from both models...' : 'Generating recipe...'}
          </p>
          <p className="loading-detail">
            Analyzing your {inventory.length} inventory items and preferences...
          </p>
        </div>
      )}

      {result && !loading && (
        <div className="results">
          {compareMode ? (
            <div className="comparison-view">
              <RecipeDisplay
                recipeData={result.base_recipe}
                title="🤖 Base Model (Llama 3.2 3B Instruct)"
              />
              <RecipeDisplay
                recipeData={result.recipe}
                title="⭐ Fine-tuned Model (Lambda-trained LoRA)"
              />
            </div>
          ) : (
            <RecipeDisplay
              recipeData={result.recipe}
              title="⭐ Generated Recipe"
            />
          )}
        </div>
      )}
    </div>
  );
}

export default RecipeGenerator;
