# Docker Playwright Setup for Crawl4AI

This guide explains how to use Crawl4AI web scraper with a Docker-based browser for environments where installing Playwright system dependencies is not possible.

## Quick Start

### 1. Start Browserless Docker Container

```bash
docker run -d \
  --name playwright-cdp \
  -p 9222:3000 \
  -e "ENABLE_API_GET=true" \
  browserless/chrome:latest
```

This starts a Chrome browser with CDP (Chrome DevTools Protocol) exposed on port 9222.

### 2. Configure Crawl4AI Scraper

```python
from akd.tools.scrapers import Crawl4AIWebScraper, Crawl4AIScraperConfig

# Create configuration
config = Crawl4AIScraperConfig(
    use_docker=True,
    playwright_cdp_url="ws://127.0.0.1:9222",
    headless=True,
    debug=False,  # Set to True for verbose output
)

# Create scraper
scraper = Crawl4AIWebScraper(config)
```

### 3. Use the Scraper

```python
from akd.tools.scrapers import ScraperToolInputSchema

# Scrape a URL
params = ScraperToolInputSchema(url="https://example.com")
result = await scraper.arun(params)

print(f"Content: {result.content}")
```

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   Your Code     │         │  Docker Container│
│                 │         │                  │
│  Crawl4AI       │ ─────→  │  Browserless     │
│  WebScraper     │ CDP/WS  │  Chrome Browser  │
│                 │  :9222  │                  │
└─────────────────┘         └──────────────────┘
```

The scraper connects to the Docker container via Chrome DevTools Protocol (CDP) over WebSocket.

## Configuration Options

### Crawl4AIScraperConfig

- `use_docker` (bool): Enable Docker browser connection (default: False)
- `playwright_cdp_url` (str): CDP WebSocket URL (default: "ws://127.0.0.1:9222")
- `headless` (bool): Run browser in headless mode (default: True)
- `browser_type` (str): Browser type - "chromium", "firefox", or "webkit" (default: "chromium")
- `debug` (bool): Enable debug logging (default: False)

## Server Deployment

For server deployment where you can't install Playwright system dependencies:

```bash
# Start Docker container on server
docker run -d \
  --name playwright-cdp \
  -p 9222:3000 \
  -e "ENABLE_API_GET=true" \
  --restart unless-stopped \
  browserless/chrome:latest

# Verify it's running
curl http://localhost:9222/json/version
```

Then configure your application:

```python
config = Crawl4AIScraperConfig(
    use_docker=True,
    playwright_cdp_url="ws://localhost:9222",  # or ws://127.0.0.1:9222
    headless=True,
)
```

## Troubleshooting

### Connection Refused

If you see "Connection reset by peer" errors:

1. Check if container is running:
   ```bash
   docker ps | grep playwright-cdp
   ```

2. Verify CDP endpoint is accessible:
   ```bash
   curl http://localhost:9222/json/version
   ```

3. Check container logs:
   ```bash
   docker logs playwright-cdp
   ```

### Port Already in Use

If port 9222 is already in use:

```bash
# Use a different port
docker run -d \
  --name playwright-cdp \
  -p 9223:3000 \
  -e "ENABLE_API_GET=true" \
  browserless/chrome:latest

# Update configuration
config = Crawl4AIScraperConfig(
    use_docker=True,
    playwright_cdp_url="ws://127.0.0.1:9223",
)
```

### Container Won't Start

Try removing existing containers:

```bash
docker rm -f playwright-cdp
docker run -d --name playwright-cdp -p 9222:3000 -e "ENABLE_API_GET=true" browserless/chrome:latest
```

## Performance Considerations

- **Container Startup**: The container takes 3-5 seconds to start
- **First Request**: First scraping request may take longer as browser initializes
- **Concurrent Requests**: Browserless supports multiple concurrent connections
- **Memory Usage**: Chrome browser uses ~200-500MB of memory

## Security

For production deployments:

1. **Network Isolation**: Run container on internal network only
2. **Authentication**: Consider adding authentication to CDP endpoint
3. **Resource Limits**: Set memory/CPU limits on the container:
   ```bash
   docker run -d \
     --name playwright-cdp \
     -p 9222:3000 \
     --memory="1g" \
     --cpus="1.0" \
     browserless/chrome:latest
   ```

## Alternative: Official Playwright Docker

You can also use Microsoft's official Playwright image, but it requires more configuration:

```bash
docker run -d \
  --name playwright-cdp \
  -p 9222:9222 \
  --cap-add=SYS_ADMIN \
  mcr.microsoft.com/playwright:latest \
  bash -c "npx playwright launch chromium --browser-remote-debugging-port=9222"
```

However, browserless/chrome is recommended as it's simpler and more reliable.

## Testing

Run the test script to verify your setup:

```bash
python test_docker_scraper.py
```

Expected output:
```
Testing Crawl4AI with Docker CDP...

Scraping: https://example.com
Docker CDP: ws://127.0.0.1:9222
[INIT].... → Crawl4AI 0.6.3
[FETCH]... ↓ https://example.com/
| ✓ | ⏱: 0.45s
[SCRAPE].. ◆ https://example.com/
| ✓ | ⏱: 0.00s
[COMPLETE] ● https://example.com/
| ✓ | ⏱: 0.46s

✓ Success!
Content length: 229 chars
```

## References

- [Browserless Chrome](https://www.browserless.io/)
- [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
- [Crawl4AI Documentation](https://docs.crawl4ai.com/)
