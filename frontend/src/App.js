import React, { useState } from 'react';
import './index.css';

function App() {
  const [formData, setFormData] = useState({
    time_spent_alone: '',
    social_event_attendance: '',
    going_outside: '',
    friends_circle_size: '',
    post_frequency: '',
    stage_fear: '',
    drained_after_socializing: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Call the real backend API
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const result = await response.json();
      setResult(result);
    } catch (err) {
      setError('Failed to get prediction. Please make sure the backend server is running.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };



  const resetForm = () => {
    setFormData({
      time_spent_alone: '',
      social_event_attendance: '',
      going_outside: '',
      friends_circle_size: '',
      post_frequency: '',
      stage_fear: '',
      drained_after_socializing: ''
    });
    setResult(null);
    setError(null);
  };

  return (
    <div className="container">
      <div className="header">
        <h1>🧠 Personality Predictor</h1>
        <p>Test your personality prediction model with this simple interface</p>
      </div>

      {error && (
        <div className="error">
          {error}
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Analyzing your personality...</p>
        </div>
      )}

      {!result && !loading && (
        <div className="form-container">
          <h2 style={{ marginBottom: '20px', color: '#333' }}>Enter Your Data</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Time Spent Alone (hours per day)</label>
              <input
                type="number"
                name="time_spent_alone"
                value={formData.time_spent_alone}
                onChange={handleInputChange}
                placeholder="e.g., 4"
                min="0"
                step="0.1"
                required
              />
            </div>

            <div className="form-group">
              <label>Social Event Attendance (per month)</label>
              <input
                type="number"
                name="social_event_attendance"
                value={formData.social_event_attendance}
                onChange={handleInputChange}
                placeholder="e.g., 8"
                min="0"
                required
              />
            </div>

            <div className="form-group">
              <label>Going Outside (times per week)</label>
              <input
                type="number"
                name="going_outside"
                value={formData.going_outside}
                onChange={handleInputChange}
                placeholder="e.g., 5"
                min="0"
                required
              />
            </div>

            <div className="form-group">
              <label>Friends Circle Size</label>
              <input
                type="number"
                name="friends_circle_size"
                value={formData.friends_circle_size}
                onChange={handleInputChange}
                placeholder="e.g., 10"
                min="0"
                required
              />
            </div>

            <div className="form-group">
              <label>Post Frequency (per week)</label>
              <input
                type="number"
                name="post_frequency"
                value={formData.post_frequency}
                onChange={handleInputChange}
                placeholder="e.g., 3"
                min="0"
                required
              />
            </div>

            <div className="form-group">
              <label>Stage Fear</label>
              <div className="radio-group">
                <label className="radio-option">
                  <input
                    type="radio"
                    name="stage_fear"
                    value="Yes"
                    checked={formData.stage_fear === 'Yes'}
                    onChange={handleInputChange}
                    required
                  />
                  Yes
                </label>
                <label className="radio-option">
                  <input
                    type="radio"
                    name="stage_fear"
                    value="No"
                    checked={formData.stage_fear === 'No'}
                    onChange={handleInputChange}
                    required
                  />
                  No
                </label>
              </div>
            </div>

            <div className="form-group">
              <label>Drained After Socializing</label>
              <div className="radio-group">
                <label className="radio-option">
                  <input
                    type="radio"
                    name="drained_after_socializing"
                    value="Yes"
                    checked={formData.drained_after_socializing === 'Yes'}
                    onChange={handleInputChange}
                    required
                  />
                  Yes
                </label>
                <label className="radio-option">
                  <input
                    type="radio"
                    name="drained_after_socializing"
                    value="No"
                    checked={formData.drained_after_socializing === 'No'}
                    onChange={handleInputChange}
                    required
                  />
                  No
                </label>
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? 'Analyzing...' : 'Get Personality Prediction'}
            </button>
          </form>
        </div>
      )}

      {result && (
        <div className="result-container">
          <h2 className="result-title">Your Personality Result</h2>
          <p className="result-confidence">
            Confidence: {result.confidence}%
          </p>
          <div className={`personality-type ${result.personality_type.toLowerCase()}`}>
            {result.personality_type}
          </div>
          <p style={{ color: '#666', marginBottom: '20px' }}>
            Based on your behavioral patterns, you are likely an {result.personality_type.toLowerCase()}.
          </p>
          <button onClick={resetForm} className="reset-btn">
            Test Again
          </button>
        </div>
      )}
    </div>
  );
}

export default App; 