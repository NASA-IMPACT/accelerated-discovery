import React, { useState, useEffect } from 'react';
import PasswordModal from './PasswordModal';

const PasswordProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    // Check if user is already authenticated in this session
    const checkAuthStatus = () => {
      const authStatus = sessionStorage.getItem('earth_data_authenticated');
      setIsAuthenticated(authStatus === 'true');
      setIsChecking(false);
    };

    // Small delay to prevent flash of content
    const timer = setTimeout(checkAuthStatus, 100);

    return () => clearTimeout(timer);
  }, []);

  const handleAuthenticate = (authenticated) => {
    setIsAuthenticated(authenticated);
  };

  // Show nothing while checking authentication status
  if (isChecking) {
    return (
      <div className="auth-loading">
        <div className="auth-loading-spinner"></div>
      </div>
    );
  }

  // Show password modal if not authenticated
  if (!isAuthenticated) {
    return <PasswordModal onAuthenticate={handleAuthenticate} />;
  }

  // Show main app if authenticated
  return children;
};

export default PasswordProvider;
