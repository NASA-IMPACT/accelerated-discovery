# Profiling Scripts

This directory contains scripts for profiling memory, CPU, and GPU usage of AKD agents using Memray and Scalene.

## Quick Start

### Option 1: Simple `memray run` (Recommended!)

The simplest way - just use memray's CLI directly on the bare scripts:

```bash
# Gap Agent - Live mode (real-time web UI at http://localhost:8080)
memray run --live scripts/profilers/profile_gap_agent.py

# Gap Agent - Save to file
memray run -o gap_agent.bin scripts/profilers/profile_gap_agent.py

# DeepLitSearchAgent - Live mode
memray run --live scripts/profilers/profile_deep_search_agent.py

# DeepLitSearchAgent - Save to file
memray run -o deep_search.bin scripts/profilers/profile_deep_search_agent.py
```

**Why this is best:**
- No code modifications needed
- Standard memray CLI
- Automatic class/method-level tracking
- Works out of the box

### Option 2: All-in-one profiler with CLI

Use the comprehensive profiler script with built-in options:

```bash
# Gap Agent with live mode
python scripts/profilers/akd_memory_profiler_memray.py --agent gap --live

# DeepLitSearchAgent
python scripts/profilers/akd_memory_profiler_memray.py --agent deep_search

# Both agents
python scripts/profilers/akd_memory_profiler_memray.py --agent both

# Custom output directory
python scripts/profilers/akd_memory_profiler_memray.py --agent gap --output-dir ./profiles
```

## Viewing Results

After running in non-live mode, analyze the `.bin` files:

```bash
# Interactive flamegraph (BEST for visual insights)
memray flamegraph gap_agent.bin

# Top allocators table
memray table gap_agent.bin

# Call tree view
memray tree gap_agent.bin

# Summary statistics
memray stats gap_agent.bin
```

## Scripts Overview

### `profile_gap_agent.py`
- Barebones Gap Agent runner
- Use with `memray run` CLI
- ~80 lines, no complexity

### `profile_deep_search_agent.py`
- Barebones DeepLitSearchAgent runner
- Use with `memray run` CLI
- ~60 lines, no complexity

### `akd_memory_profiler_memray.py`
- All-in-one profiler with CLI
- Supports both agents
- Built-in live mode support
- Programmatic Tracker() usage

## Examples

### See GapAgent class-level allocations in real-time:
```bash
memray run --live scripts/profilers/profile_gap_agent.py
```
Opens browser showing:
- `GapAgent._fetch_paper_items` allocations
- `GapAgent._fetch_parsed_pdfs` allocations
- `GapAgent.create_graph` allocations
- Call chains into third-party libraries

### Generate offline analysis:
```bash
memray run -o gap.bin scripts/profilers/profile_gap_agent.py
memray flamegraph gap.bin  # Opens interactive HTML
```

### Filter to show only your code:
```bash
memray tree gap.bin | grep gap_analysis
```

## Requirements

- `memray`: Already installed (`uv pip list | grep memray`)
- `OPENAI_API_KEY` in `.env` file
- Internet connection for fetching papers

## Tips

1. **Use `--live` for interactive debugging** - see allocations as they happen
2. **Use flamegraph for analysis** - best visualization of memory hierarchy
3. **Filter third-party libraries** - focus on your AKD code
4. **Compare runs** - profile before/after optimizations

## Option 3: Scalene - AI-Powered Profiler (CPU + Memory + GPU) ⭐ Best All-in-One

**Scalene** is a high-performance profiler that shows CPU, memory, AND GPU usage with AI-powered optimization suggestions.

### Installation
```bash
uv pip install scalene
```

### Usage - Simple and Powerful

```bash
# Profile Gap Agent with interactive HTML output
scalene scripts/profilers/profile_gap_agent.py

# Profile with reduced overhead (sampling)
scalene --reduced-profile scripts/profilers/profile_gap_agent.py

# Profile with AI optimization suggestions (requires OpenAI API key)
scalene --ai scripts/profilers/profile_gap_agent.py

# Profile only memory (no CPU)
scalene --profile-only-memory scripts/profilers/profile_gap_agent.py

# Profile only CPU (no memory)
scalene --profile-only-cpu scripts/profilers/profile_gap_agent.py

# Output to JSON for programmatic analysis
scalene --json --outfile gap_profile.json scripts/profilers/profile_gap_agent.py

# Profile with custom interval (default 0.01s)
scalene --cpu-sampling-rate 0.001 scripts/profilers/profile_gap_agent.py
```

### What Scalene Shows You

Scalene provides a rich HTML report with:

1. **Line-by-line CPU time** - precise timing for each line
2. **Memory usage per line** - native Python + C allocations
3. **Memory timeline** - growth over time
4. **GPU usage** - if you're using CUDA/GPU operations
5. **Copy volume** - how much data is being copied (performance hint)
6. **AI suggestions** - optimization recommendations (with `--ai` flag)

### Example Output

```
scripts/profilers/profile_gap_agent.py: % of time = 100.00% out of 15.32s.
  ╷    ╷       ╷       ╷       ╷       ╷
  │    │       │ Memory│       │       │
  │    │ Time  │ Python│ native│ net   │ Copy  │
  │Line│       │ peak  │ peak  │ MB    │ (MB/s)│[script path]
 ╶┼────┼───────┼───────┼───────┼───────┼───────┼────────────────
  │278 │  8%   │ 45 MB │ 12 MB │ +15   │ 234   │ paper_items, search_results = await...
  │287 │ 46%   │120 MB │ 45 MB │ +70   │ 456   │ graph = await self.create_graph(...)
  │302 │ 22%   │150 MB │ 48 MB │ +5    │ 123   │ graph = json_graph.node_link_data(...)
```

### Why Scalene is Great

✅ **All-in-one**: CPU + Memory + GPU in one tool
✅ **Line-by-line detail**: See timing and memory for each line
✅ **Low overhead**: Uses sampling instead of tracing
✅ **Beautiful output**: Interactive HTML with charts
✅ **AI suggestions**: Get optimization recommendations
✅ **No code changes**: Just run `scalene yourscript.py`
✅ **Async support**: Works with `asyncio` out of the box

### Scalene vs Memray

| Feature | Scalene | memray |
|---------|---------|--------|
| CPU profiling | ✅ | ❌ |
| Memory profiling | ✅ | ✅ |
| GPU profiling | ✅ | ❌ |
| Line-by-line | ✅ | ❌ |
| AI suggestions | ✅ | ❌ |
| Low overhead | ✅ | ✅ |
| Native code | ✅ | ✅ |
| Interactive flamegraph | ⚠️ | ✅ |
| Memory timeline | ✅ | ✅ |

### Recommended Workflow

```bash
# 1. Quick overview with Scalene (CPU + Memory)
scalene scripts/profilers/profile_gap_agent.py

# 2. Deep memory analysis with memray (if needed)
memray run --live scripts/profilers/profile_gap_agent.py

# 3. Get AI optimization suggestions
scalene --ai scripts/profilers/profile_gap_agent.py
```

### Advanced Scalene Usage

```bash
# Profile and automatically open browser
scalene --html --outfile profile.html scripts/profilers/profile_gap_agent.py

# Profile with custom memory threshold (only show lines > 10MB)
scalene --memory-threshold 10 scripts/profilers/profile_gap_agent.py

# Profile specific lines only (add @profile decorator)
scalene --profile-all scripts/profilers/profile_gap_agent.py

# Profile in reduced mode for production (lower overhead)
scalene --reduced-profile --cpu-sampling-rate 0.1 scripts/profilers/profile_gap_agent.py
```

## Profiler Comparison Table

| Use Case | Recommended Tool | Command |
|----------|------------------|---------|
| **Quick overview (CPU + Memory)** | Scalene | `scalene scripts/profilers/profile_gap_agent.py` |
| **Deep memory analysis** | memray | `memray run --live scripts/profilers/profile_gap_agent.py` |
| **Memory flamegraph** | memray | `memray run -o gap.bin scripts/profilers/profile_gap_agent.py && memray flamegraph gap.bin` |
| **AI optimization tips** | Scalene | `scalene --ai scripts/profilers/profile_gap_agent.py` |
| **Production profiling** | Scalene (reduced) | `scalene --reduced-profile scripts/profilers/profile_gap_agent.py` |
| **GPU profiling** | Scalene | `scalene scripts/profilers/profile_gap_agent.py` |

## Troubleshooting

**Can't see GapAgent methods in output?**
- Use `memray tree` to see full call hierarchy
- Search for "gap_analysis" in the output
- Remember: bulk allocations happen in libraries (transformers, docling)
- Your methods orchestrate, libraries allocate

**Live mode not opening browser?**
- Manually open http://localhost:8080
- Check firewall settings
- Use `--live-remote` for different port

**Want to profile specific methods only?**
- Modify the script to call only specific methods
- Or use the programmatic profiler with custom Tracker() placement

**Scalene showing "no samples" or empty output?**
- Your script may be running too fast - add more iterations
- Use `--cpu-sampling-rate 0.001` for finer granularity
- Check that the script actually runs (try without scalene first)

**AI suggestions not working in Scalene?**
- Set `OPENAI_API_KEY` environment variable
- Ensure you have internet connection
- Use `--ai` flag explicitly
