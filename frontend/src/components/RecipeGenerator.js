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
              <div className="card result-card">
                <h3>🤖 Base Model (Llama 3.2 3B Instruct)</h3>
                <div className="recipe-content">
                  <pre>{result.base_recipe}</pre>
                </div>
              </div>

              <div className="card result-card finetuned">
                <h3>⭐ Fine-tuned Model (Lambda-trained LoRA)</h3>
                <div className="recipe-content">
                  <pre>{result.recipe}</pre>
                </div>
              </div>
            </div>
          ) : (
            <div className="card result-card">
              <h3>⭐ Generated Recipe</h3>
              <div className="recipe-content">
                <pre>{result.recipe}</pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default RecipeGenerator;
