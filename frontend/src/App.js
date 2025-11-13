import React, { useState, useEffect } from 'react';
import './App.css';
import Inventory from './components/Inventory';
import Settings from './components/Settings';
import RecipeGenerator from './components/RecipeGenerator';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('generate');
  const [inventory, setInventory] = useState([]);
  const [preferences, setPreferences] = useState({
    dietary_restrictions: [],
    cooking_style: 'balanced',
    custom_preferences: ''
  });

  // Load inventory and preferences on mount
  useEffect(() => {
    loadInventory();
    loadPreferences();
  }, []);

  const loadInventory = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/inventory`);
      setInventory(response.data.inventory);
    } catch (error) {
      console.error('Failed to load inventory:', error);
    }
  };

  const loadPreferences = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/preferences`);
      setPreferences(response.data);
    } catch (error) {
      console.error('Failed to load preferences:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🍳 PantryPilot Recipe Generator</h1>
        <p>AI-Powered Recipe Generation from Your Pantry</p>
      </header>

      <nav className="tabs">
        <button
          className={activeTab === 'generate' ? 'active' : ''}
          onClick={() => setActiveTab('generate')}
        >
          🎯 Generate Recipe
        </button>
        <button
          className={activeTab === 'inventory' ? 'active' : ''}
          onClick={() => setActiveTab('inventory')}
        >
          📦 My Inventory
        </button>
        <button
          className={activeTab === 'settings' ? 'active' : ''}
          onClick={() => setActiveTab('settings')}
        >
          ⚙️ Settings
        </button>
      </nav>

      <main className="content">
        {activeTab === 'generate' && (
          <RecipeGenerator
            inventory={inventory}
            preferences={preferences}
          />
        )}
        {activeTab === 'inventory' && (
          <Inventory
            inventory={inventory}
            onInventoryChange={loadInventory}
          />
        )}
        {activeTab === 'settings' && (
          <Settings
            preferences={preferences}
            onPreferencesChange={loadPreferences}
          />
        )}
      </main>

      <footer>
        <p>Model: Phi-3 Mini Fine-tuned with LoRA | Running on MLX</p>
      </footer>
    </div>
  );
}

export default App;
