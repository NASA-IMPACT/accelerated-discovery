import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import PasswordProvider from './PasswordProvider';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <PasswordProvider>
      <App />
    </PasswordProvider>
  </React.StrictMode>
);
