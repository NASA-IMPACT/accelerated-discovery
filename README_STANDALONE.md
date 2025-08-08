# 🚀 Standalone Deep Search Testing Environment

A completely self-contained notebook for testing and tweaking the DeepLitSearchAgent functionality. Just clone, run, and start experimenting!

## ✨ Quick Start (3 steps)

```bash
# 1. Clone the repository
git clone https://github.com/NASA-IMPACT/accelerated-discovery.git
cd accelerated-discovery

# 2. Run the automated setup
bash scripts/setup_standalone.sh

# 3. Start the notebook
source .venv/bin/activate
jupyter lab notebooks/deep_search_standalone.ipynb
```

That's it! The notebook will handle everything else automatically.

## 📋 What You Get

### 🎯 **Fully Self-Contained Notebook**
- **Auto-Setup**: Installs all dependencies when you run it
- **Local Search Engine**: SearxNG Docker container with JSON support
- **API Integration**: Works with OpenAI or Anthropic
- **Mock Fallback**: Works even without external services

### 🔧 **Deep Customization**
- **Editable System Prompts**: 5 different agent prompts to customize
- **Parameter Tuning**: Search iterations, quality thresholds, relevancy scores
- **Real-time Testing**: Test different configurations instantly
- **Results Analysis**: Detailed visualization and metrics

### 📊 **Advanced Analysis**
- **Relevancy Tracking**: Monitor improvement across search iterations
- **Quality Metrics**: Comprehensive results analysis with plots
- **Export Functionality**: Save results for further analysis
- **Troubleshooting Guide**: Built-in help and optimization tips

## 🎮 How It Works

### The Magic Behind the Scenes

When you open the notebook, it automatically:

1. **🔍 Checks Dependencies** → Installs anything missing
2. **🔑 Asks for API Key** → Prompts you for OpenAI/Anthropic key
3. **🐳 Starts SearxNG** → Launches local Docker search engine
4. **🤖 Loads Agent** → Initializes the deep search system
5. **✅ Ready to Go!** → You can start testing immediately

### What You Can Customize

- **Research Parameters**: How many iterations, quality thresholds
- **System Prompts**: Behavior of triage, clarification, research agents
- **Search Configuration**: Relevancy scoring, content fetching
- **Test Queries**: Predefined or custom research questions

## 🚀 Features

### 🔄 **Multi-Agent Deep Search**
- **Query Triage**: Determines if clarification is needed
- **Smart Clarification**: Asks focused follow-up questions
- **Instruction Building**: Transforms queries into research briefs
- **Iterative Research**: Progressively refines search strategy
- **Research Synthesis**: Compiles comprehensive reports

### 📊 **Results Analysis**
- **Relevancy Distribution**: Histogram of result quality scores
- **Iteration Improvement**: Tracks quality gains over search cycles
- **Performance Metrics**: First-half vs second-half analysis
- **Visual Indicators**: Color-coded relevancy levels

### 🛠️ **Local Infrastructure**
- **SearxNG Search**: Privacy-focused, no-tracking local search
- **JSON API Support**: Structured search result format
- **Scientific Engines**: Google Scholar, arXiv, PubMed, Semantic Scholar
- **Docker Containerized**: Easy setup and teardown

## 📖 Usage Examples

### Basic Testing
```python
# Set your test query
CUSTOM_QUERY = "machine learning for climate prediction"

# Adjust parameters
MAX_RESEARCH_ITERATIONS = 3
QUALITY_THRESHOLD = 0.7
MIN_RELEVANCY_SCORE = 0.3

# Run the search (cell execution)
# Results appear automatically with analysis
```

### Advanced Customization
```python
# Edit system prompts directly in cells
TRIAGE_AGENT_PROMPT = """
Your custom triage behavior here...
"""

# Compare different configurations
COMPARISON_CONFIGS = {
    "Fast": DeepLitSearchAgentConfig(max_research_iterations=1),
    "Thorough": DeepLitSearchAgentConfig(max_research_iterations=5)
}
```

## 🔧 Configuration Options

### Core Parameters
- `MAX_RESEARCH_ITERATIONS` - Search refinement cycles (1-5)
- `QUALITY_THRESHOLD` - Stop condition (0.1-1.0)
- `MIN_RELEVANCY_SCORE` - Minimum result inclusion threshold
- `FULL_CONTENT_THRESHOLD` - Score for full content fetching

### Agent Behavior
- `AUTO_CLARIFY` - Ask clarifying questions automatically
- `USE_SEMANTIC_SCHOLAR` - Include academic paper searches
- `ENABLE_PER_LINK_ASSESSMENT` - Detailed relevancy scoring
- `DEBUG_MODE` - Show detailed agent reasoning

## 📈 Understanding Results

### Relevancy Metrics
- **🟢 High (≥0.7)**: Excellent topic alignment, full content fetched
- **🟡 Medium (0.3-0.7)**: Good relevance, summary only
- **🔴 Low (<0.3)**: Poor relevance, filtered out

### Iteration Analysis
- **Trend Line**: Shows if search quality improves over iterations
- **First vs Second Half**: Measures learning effectiveness
- **Distribution**: Visualizes result quality spread

### Quality Indicators
- **Research Report**: Synthesized findings from all iterations
- **Source Diversity**: Mix of academic and general sources
- **Conflict Detection**: Identification of contradictory evidence

## 🐛 Troubleshooting

### Common Issues

**SearxNG not starting:**
```bash
# Check Docker status
docker ps

# Restart container
cd searxng && docker-compose restart
```

**API key issues:**
- Verify key is correct and has credits
- Check for typos in .env file
- Try alternative provider (OpenAI ↔ Anthropic)

**No results found:**
- Lower `MIN_RELEVANCY_SCORE` to 0.1
- Increase `MAX_RESEARCH_ITERATIONS`
- Try simpler, more specific queries

**Slow performance:**
- Set `ENABLE_FULL_CONTENT_SCRAPING = False`
- Reduce iterations to 1-2 for testing
- Disable debug mode

### Getting Help

1. **Check the troubleshooting section** in the notebook
2. **Review configuration parameters** - many issues are config-related
3. **Try the quick test panel** for simplified debugging
4. **Check Docker logs**: `docker logs searxng_standalone`

## 🤝 Sharing the Environment

### For Researchers
```bash
# Share the entire setup
git clone https://github.com/NASA-IMPACT/accelerated-discovery.git
cd accelerated-discovery
bash scripts/setup_standalone.sh
# Add API keys to .env
jupyter lab notebooks/deep_search_standalone.ipynb
```

### For Developers
The notebook includes:
- Embedded simplified agent implementation
- Mock data fallbacks for testing
- Complete configuration management
- Export/import functionality

## 🔒 Privacy & Security

### Local-First Design
- **SearxNG**: No tracking, no data sharing with search engines
- **API Keys**: Stored locally in .env file, never committed
- **Docker Isolation**: Search engine runs in isolated container
- **Results Storage**: All data stays on your machine

### Best Practices
- Keep API keys in .env file, not in code
- Use SearxNG for sensitive research queries
- Export results to secure locations
- Regular Docker image updates for security

## 🎯 Perfect For

### 📚 **Researchers**
- Testing search strategies for literature reviews
- Comparing different relevancy assessment approaches
- Experimenting with multi-agent research workflows
- Validating search quality across different domains

### 🔬 **Scientists**
- Domain-specific search optimization
- Iterative research methodology testing
- Quality threshold calibration
- Custom prompt engineering for scientific domains

### 👨‍💻 **Developers**
- Understanding multi-agent search architecture
- Testing search algorithm improvements
- Benchmarking different configurations
- Prototyping custom research agents

## 🚀 Next Steps

1. **Start with the simple test queries** to understand the system
2. **Experiment with different parameters** to see their impact
3. **Edit the system prompts** to customize agent behavior
4. **Try your own research questions** in your domain
5. **Share configurations** that work well for your use case

---

**🎉 Ready to revolutionize your research process?**

Just run the three commands at the top and start exploring! The notebook will guide you through everything else.