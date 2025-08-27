import React, { useState, useEffect, useRef } from 'react';
import { Lock, AlertCircle } from 'lucide-react';

const PasswordModal = ({ onAuthenticate }) => {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    // Auto-focus the password input when modal opens
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Simulate a brief loading state for UX
    setTimeout(() => {
      const correctPassword = process.env.REACT_APP_ACCESS_PASSWORD;

      if (password === correctPassword) {
        // Store authentication in session storage (cleared when browser closes)
        sessionStorage.setItem('earth_data_authenticated', 'true');
        onAuthenticate(true);
      } else {
        setError('Incorrect password. Please try again.');
        setPassword('');
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }
      setIsLoading(false);
    }, 500);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !isLoading) {
      handleSubmit(e);
    }
  };

  return (
    <div className="password-modal-overlay">
      <div className="password-modal">
        <div className="password-modal-header">
          <div className="password-modal-icon">
            <Lock size={32} />
          </div>
          <h2 className="password-modal-title">Access Required</h2>
          <p className="password-modal-subtitle">
            Please enter the password to access the Earth Science Data Discovery tool
          </p>
        </div>

        <form className="password-modal-form" onSubmit={handleSubmit}>
          <div className="password-input-container">
            <input
              ref={inputRef}
              type="password"
              className="password-input"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
            />
          </div>

          {error && (
            <div className="password-error">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            className="password-submit-button"
            disabled={isLoading || !password.trim()}
          >
            {isLoading ? 'Verifying...' : 'Access'}
          </button>
        </form>

        <div className="password-modal-footer">
          <p className="password-modal-note">
            This password protects against automated access.
            Contact the administrator if you need access.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PasswordModal;
