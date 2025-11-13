import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Settings.css';

const API_BASE = 'http://localhost:8000';

const DIETARY_OPTIONS = [
  'vegetarian',
  'vegan',
  'gluten-free',
  'dairy-free',
  'low-carb',
  'keto',
  'paleo',
  'nut-free'
];

const COOKING_STYLES = [
  { value: 'quick', label: 'Quick & Easy' },
  { value: 'healthy', label: 'Healthy & Nutritious' },
  { value: 'comfort', label: 'Comfort Food' },
  { value: 'gourmet', label: 'Gourmet' },
  { value: 'balanced', label: 'Balanced' }
];

function Settings({ preferences, onPreferencesChange }) {
  const [localPrefs, setLocalPrefs] = useState(preferences);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setLocalPrefs(preferences);
  }, [preferences]);

  const toggleDietaryRestriction = (restriction) => {
    const current = localPrefs.dietary_restrictions || [];
    const updated = current.includes(restriction)
      ? current.filter(r => r !== restriction)
      : [...current, restriction];

    setLocalPrefs({
      ...localPrefs,
      dietary_restrictions: updated
    });
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await axios.post(`${API_BASE}/api/preferences`, localPrefs);
      onPreferencesChange();
      alert('Settings saved successfully!');
    } catch (error) {
      console.error('Failed to save preferences:', error);
      alert('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-container">
      <div className="card">
        <h2>⚙️ Your Preferences</h2>
        <p className="description">
          Customize your recipe generation preferences. These settings will be passed to the LLM as JSON.
        </p>

        <div className="settings-section">
          <h3>Dietary Restrictions</h3>
          <div className="dietary-options">
            {DIETARY_OPTIONS.map(option => (
              <label key={option} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={(localPrefs.dietary_restrictions || []).includes(option)}
                  onChange={() => toggleDietaryRestriction(option)}
                />
                <span>{option}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="settings-section">
          <h3>Cooking Style</h3>
          <select
            value={localPrefs.cooking_style || 'balanced'}
            onChange={(e) => setLocalPrefs({ ...localPrefs, cooking_style: e.target.value })}
          >
            {COOKING_STYLES.map(style => (
              <option key={style.value} value={style.value}>
                {style.label}
              </option>
            ))}
          </select>
        </div>

        <div className="settings-section">
          <h3>Custom Preferences</h3>
          <textarea
            placeholder="Add any additional preferences (e.g., 'low sodium', 'no spicy', 'family-friendly')..."
            value={localPrefs.custom_preferences || ''}
            onChange={(e) => setLocalPrefs({ ...localPrefs, custom_preferences: e.target.value })}
          />
          <p className="hint">
            These preferences will be included in the prompt sent to the LLM.
          </p>
        </div>

        <div className="settings-section">
          <h3>Preview - JSON Format</h3>
          <pre className="json-preview">
            {JSON.stringify(localPrefs, null, 2)}
          </pre>
          <p className="hint">
            This JSON will be sent to the model for recipe generation.
          </p>
        </div>

        <button
          className="primary save-button"
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}

export default Settings;
