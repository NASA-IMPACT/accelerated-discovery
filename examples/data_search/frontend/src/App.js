import React, { useState, useEffect, useCallback, useRef, useReducer } from 'react';
import { Search, Loader2, ChevronDown, ChevronRight, Download, AlertCircle } from 'lucide-react';
import { loadConfig, getConfig, getWebSocketUrl, isDevelopment } from './config';
import './App.css';

// Enhanced WebSocket hook with connection management and auto-reconnect
const useWebSocket = (searchId) => {
  const [socket, setSocket] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [messageBuffer, setMessageBuffer] = useState([]);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const reconnectTimeoutRef = useRef(null);
  const maxReconnectAttempts = 5;
  const baseReconnectDelay = 1000; // 1 second

  const connectWebSocket = useCallback((searchId) => {
    if (!searchId) return;

    setConnectionStatus('Connecting');
    const wsUrl = getWebSocketUrl(searchId);
    const ws = new WebSocket(wsUrl);

    if (isDevelopment()) {
      console.log('Connecting to WebSocket:', wsUrl);
    }

    ws.onopen = () => {
      setConnectionStatus('Connected');
      setSocket(ws);
      setReconnectAttempts(0);

      // Process any buffered messages
      if (messageBuffer.length > 0) {
        if (isDevelopment()) {
          console.log(`Processing ${messageBuffer.length} buffered messages`);
        }
        messageBuffer.forEach(message => setLastMessage(message));
        setMessageBuffer([]);
      }
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        // Validate message structure
        if (!message.type || !message.search_id) {
          if (isDevelopment()) {
            console.warn('Invalid message format:', message);
          }
          return;
        }

        // Deduplicate messages based on type and timestamp
        setLastMessage(prevMessage => {
          if (prevMessage &&
              prevMessage.type === message.type &&
              prevMessage.timestamp === message.timestamp) {
            return prevMessage; // Skip duplicate
          }
          return message;
        });
      } catch (error) {
        if (isDevelopment()) {
          console.error('Failed to parse WebSocket message:', error);
        }
      }
    };

    ws.onerror = (error) => {
      if (isDevelopment()) {
        console.error('WebSocket error:', error);
      }
      setConnectionStatus('Error');
    };

    ws.onclose = (event) => {
      setSocket(null);

      // Only attempt reconnection if it wasn't a normal close
      if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
        const delay = baseReconnectDelay * Math.pow(2, reconnectAttempts);
        if (isDevelopment()) {
          console.log(`WebSocket closed, reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);
        }

        setConnectionStatus('Reconnecting');
        setReconnectAttempts(prev => prev + 1);

        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket(searchId);
        }, delay);
      } else {
        setConnectionStatus('Disconnected');
        if (reconnectAttempts >= maxReconnectAttempts && isDevelopment()) {
          console.error('Max reconnection attempts reached');
        }
      }
    };

    return ws;
  }, [messageBuffer, reconnectAttempts]);

  useEffect(() => {
    if (!searchId) return;

    const ws = connectWebSocket(searchId);

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close(1000, 'Component unmounting');
      }
    };
  }, [searchId, connectWebSocket]);

  return {
    socket,
    lastMessage,
    connectionStatus,
    reconnectAttempts,
    isConnected: connectionStatus === 'Connected'
  };
};

// Search state reducer for atomic updates
const searchStateReducer = (state, action) => {
  switch (action.type) {
    case 'RESET_SEARCH':
      return {
        ...state,
        isSearching: false,
        error: null,
        results: {
          angles: [],
          queries: [],
          collections: [],
          granules: [],
          metadata: null
        },
        expandedSections: {
          angles: false,
          queries: false,
          collections: false,
          granules: false
        },
        progress: {
          currentStep: 0,
          totalSteps: 7,
          stepName: 'Ready',
          percentage: 0
        },
        collectionsChunking: {
          isActive: false,
          totalChunks: 0,
          receivedChunks: [],
          expectedCollections: 0
        }
      };

    case 'START_SEARCH':
      return {
        ...state,
        isSearching: true,
        error: null,
        progress: {
          currentStep: 0,
          totalSteps: 7,
          stepName: 'Initializing...',
          percentage: 0
        }
      };

    case 'SET_ERROR':
      return {
        ...state,
        isSearching: false,
        error: action.payload
      };

    case 'UPDATE_PROGRESS':
      return {
        ...state,
        progress: {
          currentStep: action.payload.currentStep,
          totalSteps: action.payload.totalSteps,
          stepName: action.payload.stepName,
          percentage: action.payload.percentage
        }
      };

    case 'SET_ANGLES':
      return {
        ...state,
        results: {
          ...state.results,
          angles: action.payload
        },
        expandedSections: {
          ...state.expandedSections,
          angles: true
        }
      };

    case 'SET_QUERIES':
      return {
        ...state,
        results: {
          ...state.results,
          queries: action.payload
        },
        expandedSections: {
          ...state.expandedSections,
          queries: true
        }
      };

    case 'SET_COLLECTIONS':
      return {
        ...state,
        results: {
          ...state.results,
          collections: action.payload
        },
        expandedSections: {
          ...state.expandedSections,
          collections: true
        }
      };

    case 'SET_GRANULES':
      return {
        ...state,
        results: {
          ...state.results,
          granules: action.payload.granules,
          metadata: action.payload.metadata || state.results.metadata
        },
        expandedSections: {
          ...state.expandedSections,
          granules: true
        }
      };

    case 'COMPLETE_SEARCH':
      return {
        ...state,
        isSearching: false
      };

    case 'TOGGLE_SECTION':
      return {
        ...state,
        expandedSections: {
          ...state.expandedSections,
          [action.payload]: !state.expandedSections[action.payload]
        }
      };

    case 'START_COLLECTIONS_CHUNKING':
      return {
        ...state,
        collectionsChunking: {
          isActive: true,
          totalChunks: action.payload.totalChunks,
          receivedChunks: [],
          expectedCollections: action.payload.totalCollections
        }
      };

    case 'ADD_COLLECTIONS_CHUNK':
      const newChunks = [...state.collectionsChunking.receivedChunks];
      newChunks[action.payload.chunkIndex] = action.payload.collections;

      // Check if we have all chunks
      const allChunksReceived = newChunks.filter(chunk => chunk !== undefined).length === state.collectionsChunking.totalChunks;

      let newState = {
        ...state,
        collectionsChunking: {
          ...state.collectionsChunking,
          receivedChunks: newChunks
        }
      };

      // If all chunks received, assemble the complete collections array
      if (allChunksReceived) {
        const allCollections = newChunks.flat();
        newState.results = {
          ...state.results,
          collections: allCollections
        };
        newState.expandedSections = {
          ...state.expandedSections,
          collections: true
        };
        newState.collectionsChunking = {
          ...state.collectionsChunking,
          isActive: false
        };
      }

      return newState;

    case 'COMPLETE_COLLECTIONS_CHUNKING':
      return {
        ...state,
        collectionsChunking: {
          ...state.collectionsChunking,
          isActive: false
        }
      };

    default:
      return state;
  }
};

const initialSearchState = {
  isSearching: false,
  error: null,
  results: {
    angles: [],
    queries: [],
    collections: [],
    granules: [],
    metadata: null
  },
  expandedSections: {
    angles: false,
    queries: false,
    collections: false,
    granules: false
  },
  progress: {
    currentStep: 0,
    totalSteps: 7,
    stepName: 'Ready',
    percentage: 0
  },
  // Collections chunking state
  collectionsChunking: {
    isActive: false,
    totalChunks: 0,
    receivedChunks: [],
    expectedCollections: 0
  }
};

// Expandable section component
const ExpandableSection = ({ title, isExpanded, onToggle, children, isLoading = false, count = null }) => {
  return (
    <div className="expandable-section">
      <button
        className="section-header"
        onClick={onToggle}
        disabled={isLoading}
      >
        <div className="section-header-content">
          {isLoading ? (
            <Loader2 className="section-icon loading" size={20} />
          ) : isExpanded ? (
            <ChevronDown className="section-icon" size={20} />
          ) : (
            <ChevronRight className="section-icon" size={20} />
          )}
          <span className="section-title">{title}</span>
          {count !== null && (
            <span className="section-count">({count})</span>
          )}
        </div>
      </button>
      {isExpanded && (
        <div className="section-content">
          {children}
        </div>
      )}
    </div>
  );
};

// Progress indicator component
const ProgressIndicator = ({ currentStep, totalSteps, stepName, percentage }) => {
  return (
    <div className="progress-indicator">
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${percentage || 0}%` }}
        />
      </div>
      <div className="progress-text">
        Step {currentStep} of {totalSteps}: {stepName}
      </div>
    </div>
  );
};

function App() {
  const [query, setQuery] = useState('');
  const [currentSearchId, setCurrentSearchId] = useState(null);
  const [searchState, dispatch] = useReducer(searchStateReducer, initialSearchState);
  const [configLoaded, setConfigLoaded] = useState(false);

  // Load configuration on app startup
  useEffect(() => {
    loadConfig().then(() => {
      setConfigLoaded(true);
      if (isDevelopment()) {
        console.log('App configuration loaded:', getConfig());
      }
    });
  }, []);

  const { lastMessage, connectionStatus, reconnectAttempts, isConnected } = useWebSocket(currentSearchId);

  // Handle incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    try {
      const { type, data = {}, progress } = lastMessage;

    // Update progress
    if (progress) {
      dispatch({
        type: 'UPDATE_PROGRESS',
        payload: {
          currentStep: progress.current_step,
          totalSteps: progress.total_steps,
          stepName: getStepName(type),
          percentage: progress.percentage
        }
      });
    }

    // Handle different message types
    switch (type) {
      case 'search_started':
        dispatch({ type: 'START_SEARCH' });
        break;

      case 'connection_established':
        if (isDevelopment()) {
          console.log('WebSocket connected:', data?.message || 'Connected');
        }
        break;

      case 'search_error':
        dispatch({ type: 'SET_ERROR', payload: data.error || 'Search failed' });
        break;

      case 'scientific_angles_generated':
        dispatch({ type: 'SET_ANGLES', payload: data.angles || [] });
        break;

      case 'cmr_queries_started':
        if (isDevelopment()) {
          console.log('CMR query generation started');
        }
        break;

      case 'cmr_queries_generated':
        dispatch({ type: 'SET_QUERIES', payload: data.queries || [] });
        break;

      case 'collections_search_started':
        if (isDevelopment()) {
          console.log('Collection search started');
        }
        break;

      case 'collections_search_completed':
        if (isDevelopment()) {
          console.log('Collection search completed:', data.total_collections_found, 'total collections found');
        }
        break;

      case 'collections_synthesis_started':
        if (isDevelopment()) {
          console.log('Collection synthesis started');
        }
        break;

      case 'collections_chunking_started':
        if (isDevelopment()) {
          console.log('📦 Collections chunking started:', data.total_collections, 'collections in', data.total_chunks, 'chunks');
        }
        dispatch({
          type: 'START_COLLECTIONS_CHUNKING',
          payload: {
            totalChunks: data.total_chunks,
            totalCollections: data.total_collections
          }
        });
        break;

      case 'collections_chunk':
        if (isDevelopment()) {
          console.log(`📋 Received chunk ${data.chunk_index + 1}/${data.total_chunks} with ${data.chunk_size} collections`);
        }
        dispatch({
          type: 'ADD_COLLECTIONS_CHUNK',
          payload: {
            chunkIndex: data.chunk_index,
            collections: data.collections
          }
        });
        break;

      case 'collections_chunking_completed':
        if (isDevelopment()) {
          console.log('✅ Collections chunking completed:', data.total_collections, 'collections total');
        }
        dispatch({ type: 'COMPLETE_COLLECTIONS_CHUNKING' });
        break;

      case 'collections_found':
        // Legacy fallback for single large message (if chunking not used)
        if (isDevelopment()) {
          console.log('🎯 Legacy collections message received:', data.collections?.length);
        }
        if (data.collections && data.collections.length > 0) {
          dispatch({ type: 'SET_COLLECTIONS', payload: data.collections });
        }
        break;

      case 'granules_search_started':
        if (isDevelopment()) {
          console.log('Granule search started');
        }
        break;

      case 'granules_found':
        dispatch({
          type: 'SET_GRANULES',
          payload: {
            granules: data.granules || [],
            metadata: null
          }
        });
        break;

      case 'search_completed':
        dispatch({ type: 'COMPLETE_SEARCH' });
        if (data.result) {
          dispatch({
            type: 'SET_GRANULES',
            payload: {
              granules: data.result.granules || [],
              metadata: data.result.search_metadata
            }
          });
        }
        break;

      default:
        if (isDevelopment()) {
          console.log('Unhandled message type:', type);
        }
    }
    } catch (error) {
      if (isDevelopment()) {
        console.error('Error processing WebSocket message:', error, lastMessage);
      }
    }
  }, [lastMessage]);

  const getStepName = (messageType) => {
    const stepNames = {
      'search_started': 'Starting search...',
      'scientific_expansion_started': 'Retrieving documents...',
      'scientific_angles_started': 'Generating scientific angles...',
      'cmr_queries_started': 'Creating search queries...',
      'collections_search_started': 'Searching collections...',
      'collections_search_completed': 'Processing search results...',
      'collections_synthesis_started': 'Filtering collections...',
      'collections_found': 'Collections ready for data search...',
      'granules_search_started': 'Finding data files...',
      'search_completed': 'Search completed!'
    };
    return stepNames[messageType] || 'Processing...';
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    try {
      dispatch({ type: 'RESET_SEARCH' });

      const response = await fetch('/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      setCurrentSearchId(result.search_id);

      // Give WebSocket time to connect before search starts
      const config = getConfig();
      setTimeout(() => {
        if (connectionStatus !== 'Connected' && isDevelopment()) {
          console.warn('WebSocket not connected when search started');
        }
      }, config.websocket.connectionTimeout / 10);  // Wait 1/10th of connection timeout

    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: `Failed to start search: ${error.message}` });
    }
  };

  const toggleSection = (section) => {
    dispatch({ type: 'TOGGLE_SECTION', payload: section });
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !searchState.isSearching) {
      handleSearch();
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌍 Earth Science Data Discovery</h1>
        <p>Find NASA Earth science datasets using natural language queries</p>
      </header>

      <main className="app-main">
        {/* Search Input */}
        <div className="search-section">
          <div className="search-input-container">
            <input
              type="text"
              className="search-input"
              placeholder="Enter your scientific query... (e.g., 'Find MODIS vegetation data over Amazon rainforest in 2023')"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={searchState.isSearching}
            />
            <button
              className="search-button"
              onClick={handleSearch}
              disabled={searchState.isSearching || !query.trim() || !configLoaded}
            >
              {searchState.isSearching ? (
                <Loader2 className="button-icon loading" size={20} />
              ) : (
                <Search className="button-icon" size={20} />
              )}
              {searchState.isSearching ? 'Searching...' : 'Search'}
            </button>
          </div>
        </div>

        {/* Progress Indicator */}
        {(searchState.isSearching || currentSearchId) && configLoaded && (
          <ProgressIndicator
            currentStep={searchState.progress.currentStep}
            totalSteps={searchState.progress.totalSteps}
            stepName={searchState.progress.stepName}
            percentage={searchState.progress.percentage}
          />
        )}

        {/* Connection Status */}
        {currentSearchId && configLoaded && (
          <div className={`connection-status status-${connectionStatus.toLowerCase()}`}>
            <div className="status-indicator">
              <span className={`status-dot ${connectionStatus.toLowerCase()}`}></span>
              Connection: {connectionStatus}
              {reconnectAttempts > 0 && (
                <span className="reconnect-info"> (Attempt {reconnectAttempts}/5)</span>
              )}
            </div>
            {!isConnected && currentSearchId && (
              <div className="connection-warning">
                ⚠️ Real-time updates may be delayed
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {searchState.error && (
          <div className="error-message">
            <AlertCircle size={20} />
            <span>{searchState.error}</span>
          </div>
        )}

        {/* Results Sections */}
        <div className="results-container">
          {/* Scientific Angles */}
          <ExpandableSection
            title="Scientific Angles"
            isExpanded={searchState.expandedSections.angles}
            onToggle={() => toggleSection('angles')}
            count={searchState.results.angles.length}
          >
            <div className="angles-grid">
              {searchState.results.angles.map((angle, index) => (
                <div key={index} className="angle-card">
                  <h4 className="angle-title">{angle.title}</h4>
                  <p className="angle-justification">{angle.scientific_justification}</p>
                </div>
              ))}
            </div>
          </ExpandableSection>

          {/* CMR Queries */}
          <ExpandableSection
            title="Search Queries"
            isExpanded={searchState.expandedSections.queries}
            onToggle={() => toggleSection('queries')}
            count={searchState.results.queries.length}
          >
            <div className="queries-list">
              {searchState.results.queries.map((query, index) => (
                <div key={index} className="query-card">
                  <div className="query-params">
                    {query.keyword && <span className="param">Keywords: {query.keyword}</span>}
                    {query.platform && <span className="param">Platform: {query.platform}</span>}
                    {query.instrument && <span className="param">Instrument: {query.instrument}</span>}
                    {query.temporal && <span className="param">Temporal: {query.temporal}</span>}
                    {query.bounding_box && <span className="param">Spatial: {query.bounding_box}</span>}
                  </div>
                </div>
              ))}
            </div>
          </ExpandableSection>

          {/* Collections */}
          <ExpandableSection
            title="Data Collections"
            isExpanded={searchState.expandedSections.collections}
            onToggle={() => toggleSection('collections')}
            count={searchState.results.collections.length}
          >
            <div className="collections-list">
              {searchState.results.collections.map((collection, index) => (
                <div key={index} className="collection-card">
                  <h4 className="collection-title">{collection.dataset_id || collection.title}</h4>
                  <p className="collection-description">{collection.summary}</p>
                  <div className="collection-metadata">
                    {collection.platform && <span className="metadata-item">Platform: {collection.platform}</span>}
                    {collection.instrument && <span className="metadata-item">Instrument: {collection.instrument}</span>}
                  </div>
                </div>
              ))}
            </div>
          </ExpandableSection>

          {/* Granules/Data Files */}
          <ExpandableSection
            title="Data Files"
            isExpanded={searchState.expandedSections.granules}
            onToggle={() => toggleSection('granules')}
            count={searchState.results.granules.length}
          >
            <div className="granules-list">
              {searchState.results.granules.map((granule, index) => (
                <div key={index} className="granule-card">
                  <div className="granule-header">
                    <h4 className="granule-title">{granule.producer_granule_id || granule.title}</h4>
                    {granule.file_size && (
                      <span className="granule-size">{granule.file_size} MB</span>
                    )}
                  </div>
                  <div className="granule-metadata">
                    {granule.time_start && (
                      <span className="metadata-item">Date: {granule.time_start}</span>
                    )}
                    {granule.online_access_flag && (
                      <span className="metadata-item access-available">Online Access Available</span>
                    )}
                  </div>
                  {granule.links && granule.links.length > 0 && (
                    <div className="granule-links">
                      {granule.links.slice(0, 3).map((link, linkIndex) => (
                        <a
                          key={linkIndex}
                          href={link.href}
                          className="download-link"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <Download size={16} />
                          Download {link.title || `File ${linkIndex + 1}`}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </ExpandableSection>
        </div>
      </main>
    </div>
  );
}

export default App;
