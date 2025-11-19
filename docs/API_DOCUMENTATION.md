# API文档和日志使用指南

## API文档

### Swagger UI (推荐)

启动服务后访问：`http://localhost:8000/docs`

Swagger UI 提供交互式API文档，支持：
- 查看所有API端点
- 查看请求/响应模型
- 在线测试API（Try it out）
- 查看示例请求和响应

### ReDoc

启动服务后访问：`http://localhost:8000/redoc`

ReDoc 提供更美观的文档展示，适合阅读和分享。

### OpenAPI Schema

访问：`http://localhost:8000/openapi.json`

获取完整的OpenAPI 3.0规范JSON文件，可用于：
- 生成客户端SDK
- API测试工具集成
- 第三方工具导入

## API标签分类

所有API按功能分为以下标签：

### 1. 数据服务
- `GET /api/v1/data/stocks` - 获取股票列表
- `GET /api/v1/data/stocks/{symbol}/kline` - 获取K线数据
- `GET /api/v1/data/stocks/{symbol}/info` - 获取股票信息

### 2. 预测服务
- `POST /api/v1/prediction/predict` - 生成价格预测
- `GET /api/v1/prediction/indicators/{symbol}` - 获取技术指标

### 3. 训练服务
- `POST /api/v1/training/start` - 启动模型训练
- `GET /api/v1/training/status/{task_id}` - 查询训练状态

### 4. 模型管理
- `GET /api/v1/models/list` - 获取模型列表
- `GET /api/v1/models/{model_id}` - 获取模型详情

### 5. 系统
- `GET /` - API根路径
- `GET /health` - 健康检查

## 响应头说明

所有API响应都包含以下自定义响应头：

| 响应头 | 说明 | 示例 |
|--------|------|------|
| `X-Request-ID` | 请求追踪ID，用于日志关联 | `a1b2c3d4-e5f6-7890-abcd-ef1234567890` |
| `X-Process-Time` | 请求处理时间（秒） | `0.123` |
| `X-RateLimit-Limit` | 速率限制 - 最大请求数 | `100` |
| `X-RateLimit-Remaining` | 速率限制 - 剩余请求数 | `95` |
| `X-RateLimit-Reset` | 速率限制 - 重置时间戳 | `1700308800` |

## 错误响应格式

所有错误响应遵循统一格式：

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述信息",
    "details": {
      "field": "具体字段",
      "reason": "详细原因"
    },
    "timestamp": "2025-11-18T10:30:00Z",
    "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }
}
```

### 常见错误码

| 错误码 | HTTP状态码 | 说明 |
|--------|-----------|------|
| `VALIDATION_ERROR` | 422 | 请求参数验证失败 |
| `VALUE_ERROR` | 400 | 参数值错误 |
| `RATE_LIMIT_EXCEEDED` | 429 | 请求频率超限 |
| `INTERNAL_SERVER_ERROR` | 500 | 服务器内部错误 |

## 日志系统

### 日志配置

在 `.env` 文件中配置：

```env
LOG_LEVEL=INFO          # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=logs/app.log   # 日志文件路径
LOG_JSON_FORMAT=true    # 是否使用JSON格式
```

### 日志级别

- **DEBUG**: 详细的调试信息
- **INFO**: 一般信息（默认）
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

### JSON日志格式

所有日志以JSON格式输出，便于日志分析工具处理：

```json
{
  "timestamp": "2025-11-18T08:20:46.668618Z",
  "level": "INFO",
  "logger": "src.api.main",
  "message": "请求完成: GET /api/v1/data/stocks - 状态码: 200 - 耗时: 0.123s",
  "module": "middleware",
  "function": "dispatch",
  "line": 156,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "method": "GET",
  "path": "/api/v1/data/stocks",
  "status_code": 200,
  "process_time": "0.123s",
  "client_ip": "127.0.0.1"
}
```

### 日志字段说明

| 字段 | 说明 |
|------|------|
| `timestamp` | 日志时间（UTC） |
| `level` | 日志级别 |
| `logger` | 日志器名称 |
| `message` | 日志消息 |
| `module` | 模块名 |
| `function` | 函数名 |
| `line` | 行号 |
| `request_id` | 请求追踪ID（如果有） |
| `exception` | 异常堆栈（如果有） |

### 查看日志

**实时查看日志：**
```bash
# Linux/Mac
tail -f logs/app.log

# Windows PowerShell
Get-Content logs/app.log -Wait -Tail 50
```

**搜索特定请求的日志：**
```bash
# 使用 jq 工具（需要安装）
cat logs/app.log | jq 'select(.request_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890")'

# 使用 grep
grep "a1b2c3d4-e5f6-7890-abcd-ef1234567890" logs/app.log
```

**按日志级别过滤：**
```bash
cat logs/app.log | jq 'select(.level == "ERROR")'
```

## 请求追踪

### 使用请求追踪ID

每个API请求都会自动生成一个唯一的请求追踪ID（UUID格式），用于：

1. **关联日志**：同一请求的所有日志都包含相同的 `request_id`
2. **问题排查**：通过 `request_id` 快速定位问题
3. **性能分析**：追踪请求的完整处理流程

### 自定义请求ID

客户端可以在请求头中提供自定义的请求ID：

```bash
curl -H "X-Request-ID: my-custom-id-123" http://localhost:8000/api/v1/data/stocks
```

响应头会返回相同的请求ID：

```
X-Request-ID: my-custom-id-123
```

### 日志中的请求追踪

所有与请求相关的日志都会包含 `request_id` 字段：

```json
{
  "timestamp": "2025-11-18T08:20:46.668618Z",
  "level": "INFO",
  "message": "请求开始: GET /api/v1/data/stocks",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  ...
}
```

## API限流

### 限流规则

- **限制**：100次请求/分钟（基于客户端IP）
- **响应头**：
  - `X-RateLimit-Limit`: 限制数量
  - `X-RateLimit-Remaining`: 剩余请求数
  - `X-RateLimit-Reset`: 重置时间戳

### 超限响应

当请求超过限流时，返回 429 状态码：

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "请求过于频繁，请在 60 秒后重试",
    "timestamp": "2025-11-18T10:30:00Z"
  }
}
```

## 测试API文档和日志

运行测试脚本：

```bash
python test_api_docs.py
```

测试内容：
- ✓ JSON格式日志输出
- ✓ 日志文件创建
- ✓ 请求追踪ID
- ✓ API文档配置
- ✓ 自定义响应头

## 启动服务

```bash
# 开发模式（自动重载）
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

启动后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

## 最佳实践

### 1. 使用请求追踪ID

在客户端保存响应头中的 `X-Request-ID`，遇到问题时提供给技术支持。

### 2. 监控响应时间

关注 `X-Process-Time` 响应头，识别慢请求。

### 3. 遵守限流规则

监控 `X-RateLimit-Remaining`，避免触发限流。

### 4. 处理错误响应

始终检查响应状态码，解析错误响应中的 `error` 对象。

### 5. 日志分析

使用日志分析工具（如 ELK Stack）处理JSON格式日志，进行：
- 性能分析
- 错误监控
- 用户行为分析
- 系统健康监控

## 相关文档

- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [OpenAPI规范](https://swagger.io/specification/)
- [Python日志最佳实践](https://docs.python.org/3/howto/logging.html)
