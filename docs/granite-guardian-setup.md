# Granite Guardian Setup Guide

This guide walks you through setting up Ollama and the Granite Guardian model for use with the AKD framework's content validation features.

## Prerequisites

- macOS, Linux, or Windows with WSL2
- At least 8GB of available RAM (16GB recommended)
- ~5GB of available disk space for the model

## Step 1: Install Ollama

### macOS

```bash
# Using Homebrew
brew install ollama

# Or download from the official website
curl -fsSL https://ollama.com/install.sh | sh
```

### Linux

```bash
# Install with the official script
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows (WSL2)

```bash
# Inside WSL2 terminal
curl -fsSL https://ollama.com/install.sh | sh
```

### Verify Installation

```bash
# Check if Ollama is installed
ollama --version

# Start Ollama service (if not already running)
ollama serve
```

## Step 2: Download Granite Guardian Model

The Granite Guardian model comes in two variants. Choose based on your needs:

### Option A: Granite Guardian 8B (Recommended)

```bash
# Download the 8B model (~4.7GB)
ollama pull ibm-granite/granite-guardian-8b
```

### Option B: Granite Guardian 3B (Lighter Alternative)

```bash
# Download the 3B model (~2GB)
ollama pull ibm-granite/granite-guardian-3b
```

### Verify Model Installation

```bash
# List installed models
ollama list

# You should see output like:
# NAME                                    ID              SIZE    MODIFIED
# ibm-granite/granite-guardian-8b:latest  abc123def456    4.7 GB  2 minutes ago
```

## Step 3: Configure AKD Framework

The AKD framework uses the default Ollama configuration. By default, Ollama runs on `http://localhost:11434`.

If you need to use a different Ollama server address, you can set the `OLLAMA_HOST` environment variable (this is an Ollama client configuration, not specific to AKD):

```bash
# Optional: Only if your Ollama server runs on a different address
export OLLAMA_HOST=http://your-ollama-server:11434
```

### Test the Configuration

Create a simple test script to verify the setup:

```python
import asyncio
from akd.tools.granite_guardian_tool import (
    GraniteGuardianTool,
    GraniteGuardianToolConfig,
    GraniteGuardianInputSchema,
    GuardianModelID,
)

async def test_guardian():
    # Initialize the tool
    tool = GraniteGuardianTool(
        config=GraniteGuardianToolConfig(
            model=GuardianModelID.GUARDIAN_8B
        )
    )
    
    # Test with a simple input
    test_input = GraniteGuardianInputSchema(
        query="What is the weather today?",
        risk_type="jailbreak"
    )
    
    result = await tool.arun(test_input)
    print(f"Risk assessment: {result.risk_results}")

# Run the test
asyncio.run(test_guardian())
```

## Step 4: Using Granite Guardian in Your Code

### Using the add_guardrails Decorator

```python
from akd.agents import (
    add_guardrails, 
    InstructorBaseAgent, 
    GrandrailsAgentConfig,
    GuardrailedInstructorBaseAgent,
    GuardrailedLangBaseAgent,
)
from akd.tools.granite_guardian_tool import RiskDefinition

# Method 1: Use decorator with custom guardrails
GuardrailedAgent = add_guardrails(
    input_guardrails=[RiskDefinition.JAILBREAK, RiskDefinition.HARM],
    output_guardrails=[RiskDefinition.ANSWER_RELEVANCE, RiskDefinition.GROUNDEDNESS]
)(InstructorBaseAgent)

agent = GuardrailedAgent(config=my_config)

# Method 2: Use decorator with full config
config = GrandrailsAgentConfig(
    enabled=True,
    input_risk_types=[RiskDefinition.JAILBREAK, RiskDefinition.HARM],
    output_risk_types=[RiskDefinition.ANSWER_RELEVANCE, RiskDefinition.GROUNDEDNESS],
    fail_on_risk=False  # Log warnings instead of raising exceptions
)

GuardrailedAgent = add_guardrails(config=config)(InstructorBaseAgent)
agent = GuardrailedAgent()

# Method 3: Use pre-built convenience classes
agent = GuardrailedInstructorBaseAgent()  # Uses default guardrails

# Available risk types:
# RiskDefinition.HARM, RiskDefinition.JAILBREAK, RiskDefinition.UNETHICAL_BEHAVIOR
# RiskDefinition.VIOLENCE, RiskDefinition.SOCIAL_BIAS, RiskDefinition.PROFANITY
# RiskDefinition.SEXUAL_CONTENT, RiskDefinition.GROUNDEDNESS, RiskDefinition.RELEVANCE
# RiskDefinition.ANSWER_RELEVANCE
```

### Direct Tool Usage

```python
from akd.tools.granite_guardian_tool import GraniteGuardianTool

# Use Granite Guardian as a standalone tool
guardian_tool = GraniteGuardianTool()
result = await guardian_tool.arun({
    "query": "User question here",
    "response": "Agent response here",
    "risk_type": "answer_relevance"
})
```

## Troubleshooting

### Common Issues

1. **"Ollama service not running"**

   ```bash
   # Start Ollama service
   ollama serve
   
   # Or run in background
   nohup ollama serve > /dev/null 2>&1 &
   ```

2. **"Model not found"**

   ```bash
   # Re-pull the model
   ollama pull ibm-granite/granite-guardian-8b
   
   # Check if model exists
   ollama list | grep granite
   ```

3. **"Connection refused" errors**
   - Verify Ollama is running: `curl http://localhost:11434/api/tags`
   - Check firewall settings
   - Ensure Ollama is accessible at the default address or set OLLAMA_HOST

4. **"Out of memory" errors**
   - Try the smaller 3B model instead
   - Increase system swap space
   - Close other memory-intensive applications

### Performance Tips

- **Model Loading**: First run will be slower as the model loads into memory
- **Batch Processing**: Process multiple validations together for efficiency
- **Model Selection**: Use 3B model for faster responses if 8B is too slow

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Granite Guardian Model Card](https://huggingface.co/ibm-granite/granite-guardian-8b)
- [AKD Granite Guardian Implementation](../akd/tools/granite_guardian_tool.py)
