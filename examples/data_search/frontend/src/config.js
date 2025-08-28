/**
 * Frontend configuration management
 */

let appConfig = null;

const defaultConfig = {
  environment: 'development',
  websocket: {
    url: 'ws://localhost:8003',
    maxReconnectAttempts: 5,
    baseReconnectDelay: 1000,
    connectionTimeout: 10000
  },
  search: {
    maxCollections: 10,
    enableParallelSearch: true
  }
};

/**
 * Load configuration from backend
 */
export async function loadConfig() {
  try {
    const response = await fetch('/config');
    if (response.ok) {
      appConfig = await response.json();
      // Only log in development - check the loaded config's environment
      if (appConfig?.environment === 'development') {
        console.log('Loaded configuration:', appConfig);
      }
    } else {
      console.warn('Failed to load config from backend, using defaults');
      appConfig = defaultConfig;
    }
  } catch (error) {
    console.warn('Error loading config:', error, 'Using defaults');
    appConfig = defaultConfig;
  }
  return appConfig;
}

/**
 * Get current configuration
 */
export function getConfig() {
  return appConfig || defaultConfig;
}

/**
 * Get WebSocket URL for a search ID
 */
export function getWebSocketUrl(searchId) {
  const config = getConfig();

  // Use same-origin in production/staging (no port in URL)
  const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

  if (!isLocal) {
    // Production/staging: use same-origin WebSocket behind reverse proxy
    const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${wsProto}://${window.location.host}/ws/${searchId}`;
  }

  // Local development: use configured WebSocket URL
  return `${config.websocket.url}/ws/${searchId}`;
}

/**
 * Check if running in development mode
 */
export function isDevelopment() {
  const config = getConfig();
  return config.environment === 'development';
}
