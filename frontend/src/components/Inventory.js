import React, { useState } from 'react';
import axios from 'axios';
import './Inventory.css';

const API_BASE = 'http://localhost:8000';

function Inventory({ inventory, onInventoryChange }) {
  const [newItem, setNewItem] = useState({
    name: '',
    quantity: '',
    unit: ''
  });

  const handleAddItem = async (e) => {
    e.preventDefault();

    if (!newItem.name) {
      alert('Please enter item name');
      return;
    }

    try {
      await axios.post(`${API_BASE}/api/inventory`, {
        name: newItem.name,
        quantity: parseFloat(newItem.quantity) || null,
        unit: newItem.unit || null
      });

      setNewItem({ name: '', quantity: '', unit: '' });
      onInventoryChange();
    } catch (error) {
      console.error('Failed to add item:', error);
      alert('Failed to add item');
    }
  };

  const handleRemoveItem = async (itemName) => {
    try {
      await axios.delete(`${API_BASE}/api/inventory/${encodeURIComponent(itemName)}`);
      onInventoryChange();
    } catch (error) {
      console.error('Failed to remove item:', error);
      alert('Failed to remove item');
    }
  };

  return (
    <div className="inventory-container">
      <div className="card">
        <h2>📦 My Pantry Inventory</h2>

        <form onSubmit={handleAddItem} className="add-item-form">
          <div className="form-row">
            <input
              type="text"
              placeholder="Item name (e.g., chicken breast)"
              value={newItem.name}
              onChange={(e) => setNewItem({ ...newItem, name: e.target.value })}
            />
            <input
              type="number"
              placeholder="Quantity"
              step="0.1"
              value={newItem.quantity}
              onChange={(e) => setNewItem({ ...newItem, quantity: e.target.value })}
            />
            <input
              type="text"
              placeholder="Unit (e.g., lbs)"
              value={newItem.unit}
              onChange={(e) => setNewItem({ ...newItem, unit: e.target.value })}
            />
            <button type="submit" className="primary">Add Item</button>
          </div>
        </form>
      </div>

      <div className="card">
        <h3>Current Inventory ({inventory.length} items)</h3>

        {inventory.length === 0 ? (
          <p className="empty-state">No items in inventory. Add some ingredients above!</p>
        ) : (
          <div className="inventory-grid">
            {inventory.map((item, index) => (
              <div key={index} className="inventory-item">
                <div className="item-info">
                  <strong>{item.name}</strong>
                  {item.quantity && item.unit && (
                    <span className="item-quantity">
                      {item.quantity} {item.unit}
                    </span>
                  )}
                </div>
                <button
                  className="danger"
                  onClick={() => handleRemoveItem(item.name)}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default Inventory;
