# AKD Backend Framework Implementation Plan

## Executive Summary

This document outlines the implementation plan for the AKD Backend Framework, designed to provide a robust, scalable, and user-friendly backend for the Accelerated Knowledge Discovery system. The plan leverages existing components from the AKD codebase while introducing necessary infrastructure for multi-user support, session management, and workflow persistence.

## Architecture Overview

### Core Principles
- **Maximize Reuse**: Leverage existing AKD components (state management, node templates, LangGraph integration)
- **LangGraph-Native**: Build on LangGraph's latest features for workflow orchestration
- **Scientific Integrity**: Maintain the framework's commitment to attribution, validation, and human oversight
- **API-First Design**: Clear separation between backend services and frontend consumption
- **Scalable Architecture**: Support for horizontal scaling and multi-tenant isolation

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Frontend/UI                             │
├─────────────────────────────────────────────────────────────────┤
│                      API Gateway Layer                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   FastAPI   │  │  WebSocket   │  │  Authentication   │    │
│  │  REST API   │  │   Server     │  │    Middleware     │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                      Service Layer                               │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Workflow   │  │    State     │  │      User         │    │
│  │  Service    │  │   Service    │  │    Service        │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                 LangGraph Orchestration Layer                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ StateGraph  │  │ Checkpointer │  │   Interrupt       │    │
│  │  Manager    │  │   (Redis)    │  │    Handler        │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                    Existing AKD Core                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │    Node     │  │   Agents &   │  │     State         │    │
│  │  Templates  │  │    Tools     │  │   Management      │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                    Persistence Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ PostgreSQL  │  │    Redis     │  │   Object Store    │    │
│  │  (Primary)  │  │   (Cache)    │  │    (S3/Minio)    │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Project Setup
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py         # Authentication endpoints
│   │   │   ├── workflows.py    # Workflow CRUD endpoints
│   │   │   ├── agents.py       # Agent execution endpoints
│   │   │   ├── sessions.py     # Session management
│   │   │   └── health.py       # Health check endpoints
│   │   └── dependencies.py     # Shared dependencies
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Settings management
│   │   ├── security.py         # Auth utilities
│   │   └── exceptions.py       # Custom exceptions
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py             # Database base classes
│   │   ├── session.py          # Session management
│   │   └── repositories/       # Repository pattern
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py             # User model
│   │   ├── workflow.py         # Workflow model
│   │   ├── session.py          # Session model
│   │   └── schemas/            # Pydantic schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth.py             # Authentication service
│   │   ├── workflow.py         # Workflow orchestration
│   │   ├── state.py            # State persistence
│   │   └── llm_gateway.py      # LLM routing service
│   ├── langgraph_integration/
│   │   ├── __init__.py
│   │   ├── checkpointer.py     # Custom checkpointer
│   │   ├── graphs.py           # Graph builders
│   │   └── executors.py        # Graph execution
│   └── websocket/
│       ├── __init__.py
│       ├── manager.py          # Connection management
│       └── handlers.py         # Message handlers
├── tests/
├── alembic/                    # Database migrations
├── docker/
├── requirements.txt
└── README.md
```

#### 1.2 Core Models and Schemas

**User Model** (SQLAlchemy + Pydantic):
```python
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID, primary_key=True, default=uuid4)
    email = Column(String, unique=True, nullable=True)
    username = Column(String, unique=True, nullable=False)
    is_guest = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    # API limits
    token_limit = Column(Integer, default=100000)
    tokens_used = Column(Integer, default=0)
    
    # Relationships
    sessions = relationship("Session", back_populates="user")
    workflows = relationship("Workflow", back_populates="user")
```

**Session Model**:
```python
class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True)  # Session token
    user_id = Column(UUID, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
```

**Workflow Model**:
```python
class Workflow(Base):
    __tablename__ = "workflows"
    
    id = Column(UUID, primary_key=True, default=uuid4)
    user_id = Column(UUID, ForeignKey("users.id"))
    name = Column(String)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    # Workflow state
    state_snapshot = Column(JSON)  # Serialized GlobalWorkflowState
    langgraph_thread_id = Column(String, unique=True)
    status = Column(Enum(WorkflowStatus))
    
    # Sharing
    is_public = Column(Boolean, default=False)
    fork_from_id = Column(UUID, ForeignKey("workflows.id"), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="workflows")
    checkpoints = relationship("WorkflowCheckpoint", back_populates="workflow")
```

### Phase 2: Authentication & Session Management (Week 2-3)

#### 2.1 Authentication Service
- **JWT-based authentication** with refresh tokens
- **Guest user support** with automatic account creation
- **OAuth2 integration** preparation (GitHub, Google)
- **API key management** for programmatic access

```python
class AuthService:
    async def register_user(self, email: str, password: str) -> User:
        """Register new user with email/password"""
        
    async def create_guest_user(self) -> User:
        """Create anonymous guest user"""
        
    async def login(self, username: str, password: str) -> TokenResponse:
        """Authenticate and return JWT tokens"""
        
    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token"""
        
    async def get_current_user(self, token: str) -> User:
        """Validate token and return user"""
```

#### 2.2 Session Management
- **Redis-backed session store** for performance
- **Session-to-workflow mapping**
- **Automatic session cleanup**
- **Multi-device support**

### Phase 3: LangGraph Integration (Week 3-4)

#### 3.1 Custom Checkpointer
Extend LangGraph's checkpointing for persistence:

```python
from langgraph.checkpoint.base import BaseCheckpointSaver
import redis.asyncio as redis

class RedisCheckpointer(BaseCheckpointSaver):
    """Redis-backed checkpointer for LangGraph"""
    
    def __init__(self, redis_client: redis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl
    
    async def aget(self, config: Config) -> Optional[Checkpoint]:
        """Retrieve checkpoint from Redis"""
        
    async def aput(self, config: Config, checkpoint: Checkpoint) -> None:
        """Store checkpoint in Redis with TTL"""
        
    async def alist(self, config: Config) -> List[CheckpointTuple]:
        """List available checkpoints"""
```

#### 3.2 Workflow Orchestration Service
Leverage existing AKD components:

```python
class WorkflowService:
    def __init__(self, checkpointer: RedisCheckpointer):
        self.checkpointer = checkpointer
        
    async def create_workflow(
        self, 
        user_id: UUID, 
        workflow_type: str,
        config: dict
    ) -> Workflow:
        """Create new workflow with LangGraph StateGraph"""
        
        # Reuse existing GlobalWorkflowState
        initial_state = GlobalWorkflowState(
            workflow_id=str(uuid4()),
            workflow_config=config,
            user_preferences={"user_id": str(user_id)}
        )
        
        # Build graph using existing nodes
        graph = self._build_graph(workflow_type, config)
        
        # Compile with checkpointer
        app = graph.compile(checkpointer=self.checkpointer)
        
        return workflow
    
    async def execute_step(
        self, 
        workflow_id: UUID,
        user_input: dict
    ) -> WorkflowUpdate:
        """Execute next workflow step"""
        
    async def interrupt_workflow(
        self,
        workflow_id: UUID,
        human_response: dict
    ) -> WorkflowUpdate:
        """Handle human-in-the-loop interactions"""
```

### Phase 4: Real-time Updates & API (Week 4-5)

#### 4.1 WebSocket Implementation
```python
class WorkflowWebSocketManager:
    def __init__(self):
        self.active_connections: Dict[UUID, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, workflow_id: UUID):
        """Connect client to workflow updates"""
        
    async def broadcast_update(self, workflow_id: UUID, update: dict):
        """Send update to all connected clients"""
        
    async def handle_message(self, workflow_id: UUID, message: dict):
        """Process incoming WebSocket messages"""
```

#### 4.2 RESTful API Endpoints

**Authentication Endpoints**:
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/guest` - Create guest session
- `POST /api/v1/auth/refresh` - Refresh token
- `POST /api/v1/auth/logout` - Logout

**Workflow Endpoints**:
- `GET /api/v1/workflows` - List user workflows
- `POST /api/v1/workflows` - Create workflow
- `GET /api/v1/workflows/{id}` - Get workflow details
- `PUT /api/v1/workflows/{id}` - Update workflow
- `DELETE /api/v1/workflows/{id}` - Delete workflow
- `POST /api/v1/workflows/{id}/execute` - Execute workflow step
- `POST /api/v1/workflows/{id}/interrupt` - Handle insterruption
- `GET /api/v1/workflows/{id}/export` - Export workflow
- `POST /api/v1/workflows/fork` - Fork workflow

**Agent Endpoints**:
- `GET /api/v1/agents` - List available agents
- `POST /api/v1/agents/{type}/execute` - Execute specific agent
- `GET /api/v1/agents/{type}/schema` - Get agent input/output schema

### Phase 5: Advanced Features (Week 5-6)

#### 5.1 LLM Gateway Service
Intelligent routing and rate limiting:

```python
class LLMGatewayService:
    def __init__(self, providers: List[LLMProvider]):
        self.providers = providers
        self.usage_tracker = UsageTracker()
    
    async def route_request(
        self,
        user_id: UUID,
        request: LLMRequest
    ) -> LLMResponse:
        """Route to appropriate provider based on:
        - User tier/limits
        - Model availability
        - Cost optimization
        - Load balancing
        """
        
    async def track_usage(
        self,
        user_id: UUID,
        tokens: int,
        model: str
    ):
        """Track token usage per user"""
```

#### 5.2 Workflow Sharing & Forking
```python
class WorkflowSharingService:
    async def share_workflow(
        self,
        workflow_id: UUID,
        visibility: str = "public"
    ) -> ShareableLink:
        """Generate shareable workflow link"""
        
    async def fork_workflow(
        self,
        source_id: UUID,
        user_id: UUID
    ) -> Workflow:
        """Create fork of existing workflow"""
```

## Key Implementation Details

### 1. State Persistence Strategy
- **In-Memory**: Redis for active workflow states (TTL-based)
- **Long-term**: PostgreSQL for workflow snapshots
- **Checkpoints**: Redis with periodic PostgreSQL backup
- **Large Objects**: S3/MinIO for datasets and artifacts

### 2. Security Considerations
- **JWT tokens** with short expiration (15 min) and refresh tokens
- **Rate limiting** per user/endpoint
- **Input validation** using Pydantic schemas
- **SQL injection prevention** via SQLAlchemy ORM
- **CORS configuration** for frontend integration

### 3. Scalability Approach
- **Horizontal scaling** via load balancer
- **Redis Cluster** for session/cache distribution
- **Database read replicas** for query scaling
- **Async processing** with Celery for long tasks
- **WebSocket scaling** via Redis Pub/Sub

### 4. Monitoring & Observability
- **Structured logging** with correlation IDs
- **OpenTelemetry** integration for tracing
- **Prometheus metrics** for monitoring
- **Health check endpoints** for all services
- **Error tracking** with Sentry integration

## Migration Path from Existing Code

### 1. Minimal Changes to Core AKD
- Keep all existing agent/tool/node code unchanged
- Add adapter layers for backend integration
- Extend configuration for multi-tenant support

### 2. State Adapter Pattern
```python
class StateAdapter:
    """Adapt between AKD GlobalWorkflowState and backend models"""
    
    @staticmethod
    def to_db_model(state: GlobalWorkflowState) -> dict:
        """Convert to JSON for database storage"""
        
    @staticmethod
    def from_db_model(data: dict) -> GlobalWorkflowState:
        """Reconstruct from database"""
```

### 3. Progressive Enhancement
- Start with basic workflow execution
- Add features incrementally
- Maintain backward compatibility

## Testing Strategy

### 1. Unit Tests
- Service layer logic
- State serialization/deserialization
- Authentication flows

### 2. Integration Tests
- API endpoint testing
- Database operations
- LangGraph workflow execution

### 3. End-to-End Tests
- Complete user workflows
- WebSocket communication
- Multi-user scenarios

## Deployment Architecture

### 1. Containerization
```dockerfile
# Multi-stage Dockerfile
FROM python:3.12-slim as builder
# Build dependencies

FROM python:3.12-slim
# Runtime with minimal footprint
```

### 2. Docker Compose Development
```yaml
services:
  backend:
    build: .
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:15
    
  redis:
    image: redis:7-alpine
    
  minio:
    image: minio/minio
```

### 3. Kubernetes Production
- Deployment manifests
- Service definitions
- ConfigMaps for configuration
- Secrets for sensitive data
- HPA for auto-scaling

## Timeline and Milestones

### Week 1-2: Foundation
- [ ] Project structure setup
- [ ] Database models and migrations
- [ ] Basic FastAPI application
- [ ] Authentication system

### Week 3-4: Core Features
- [ ] LangGraph integration
- [ ] Workflow execution
- [ ] State persistence
- [ ] WebSocket support

### Week 5-6: Advanced Features
- [ ] LLM gateway
- [ ] Workflow sharing
- [ ] Export functionality
- [ ] Monitoring setup

### Week 7-8: Testing & Documentation
- [ ] Comprehensive testing
- [ ] API documentation
- [ ] Deployment guides
- [ ] Performance optimization

## Success Criteria

1. **Functional Requirements**
   - Users can register, login, and maintain sessions
   - Workflows persist across sessions
   - Real-time updates via WebSocket
   - Human-in-the-loop interactions work seamlessly

2. **Performance Requirements**
   - API response time < 200ms (p95)
   - WebSocket latency < 50ms
   - Support 1000 concurrent users
   - 99.9% uptime SLA

3. **Security Requirements**
   - Secure authentication/authorization
   - Data isolation between users
   - Rate limiting enforced
   - Audit logging implemented

## Risks and Mitigations

1. **Risk**: LangGraph API changes
   - **Mitigation**: Abstract LangGraph integration, pin versions

2. **Risk**: State serialization complexity
   - **Mitigation**: Comprehensive testing, versioned schemas

3. **Risk**: WebSocket scaling issues
   - **Mitigation**: Redis Pub/Sub, sticky sessions

4. **Risk**: Token cost overruns
   - **Mitigation**: Per-user limits, cost tracking, alerts

## Conclusion

This implementation plan provides a clear path to building a production-ready backend for the AKD framework. By leveraging existing components and following best practices, we can deliver a robust, scalable solution that maintains the framework's commitment to scientific integrity while enabling multi-user collaboration and workflow persistence.