# PocketOption API- By ChipaDevTeam - Modified by Six <3

## Support us
Join PocketOption with Six's affiliate link: [Six PocketOption Affiliate link](https://u3.shortink.io/main?utm_campaign=821725&utm_source=affiliate&utm_medium=sr&a=IqeAmBtFTrEWbh&ac=api&code=DLN960)
<br>
Join PocketOption with Chipas affiliate link: [Chipas PocketOption Affiliate link](https://u3.shortink.io/smart/SDIaxbeamcYYqB) 

A comprehensive, modern async Python API for PocketOption trading platform with advanced features including persistent connections, monitoring, and extensive testing frameworks.

## Key Features

### Enhanced Connection Management
- **Complete SSID Format Support**: Works with full authentication strings from browser (format: `42["auth",{"session":"...","isDemo":1,"uid":...,"platform":1}]`)
- **Persistent Connections**: Automatic keep-alive with 20-second ping intervals (like the original API)
- **Auto-Reconnection**: Intelligent reconnection with multiple region fallback
- **Connection Pooling**: Optimized connection management for better performance

### Advanced Monitoring & Diagnostics
- **Real-time Monitoring**: Connection health, performance metrics, and error tracking
- **Diagnostics Reports**: Comprehensive health assessments with recommendations
- **Performance Analytics**: Response times, throughput analysis, and bottleneck detection
- **Alert System**: Automatic alerts for connection issues and performance problems

### Comprehensive Testing Framework
- **Load Testing**: Concurrent client simulation and stress testing
- **Integration Testing**: End-to-end validation of all components
- **Performance Benchmarks**: Automated performance analysis and optimization
- **Advanced Test Suites**: Edge cases, error scenarios, and long-running stability tests

### Performance Optimizations
- **Message Batching**: Efficient message queuing and processing
- **Concurrent Operations**: Parallel API calls for better throughput
- **Caching System**: Intelligent caching with TTL for frequently accessed data
- **Rate Limiting**: Built-in protection against API rate limits

### Robust Error Handling
- **Graceful Degradation**: Continues operation despite individual failures
- **Automatic Recovery**: Self-healing connections and operations
- **Comprehensive Logging**: Detailed error tracking and debugging information
- **Exception Management**: Type-specific error handling and recovery strategies

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd PocketOptionAPI

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Quick Start

### Getting Your SSID

To use the API with real data, you need to extract your session ID from the browser:

1. **Open PocketOption in your browser**
2. **Open Developer Tools (F12)**
3. **Go to Network tab**
4. **Filter by WebSocket (WS)**
5. **Look for authentication message starting with `42["auth"`**
6. **Copy the complete message including the `42["auth",{...}]` format**

Example SSID format:
```
42["auth",{"session":"abcd1234efgh5678","isDemo":1,"uid":12345,"platform":1}]
```

If you are unable to find it, try running the automatic SSID scraper under the `tools` folder.