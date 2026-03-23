# Travel Agent

一个面向旅行场景的智能体项目，整合了：

- `FastAPI` 后端接口
- `Streamlit` 前端工作台
- 基于 `Chroma` 的私有知识库检索
- 基于 `LangGraph` 的多路由对话编排
- 票务查询、路线规划、RAG 问答、通用建议等多类能力

这个仓库已经不是单一的 RAG Demo，而是一个“旅行助手 + 文档知识库 + 多智能体路由”的完整原型。

## 项目能力

当前代码中已经落地的核心能力包括：

- 用户注册、登录、个人资料查询、修改密码
- 文档上传、解析、写入个人向量库
- 基于个人知识库的问答
- 对旅行相关问题进行自动路由
- 针对不同问题选择不同处理链路：
  - `ticket`：票务类查询
  - `roadmap`：路线/游玩规划
  - `rag`：知识库检索问答
  - `other`：通用旅行建议
- 对需要确认的任务增加确认流程
- 对图编排结果做性能和效果评估

## 目录结构

```text
.
|-- backend/                FastAPI 后端
|   |-- app/
|   |   |-- api/routes/     鉴权、聊天、向量库接口
|   |   |-- graphs/         LangGraph 对话编排
|   |   |-- core/           数据库初始化
|   |   |-- models/         数据模型
|   |   |-- schemas/        请求/响应模型
|   |   `-- crued/          业务逻辑
|   `-- tests/              后端测试
|-- web/                    Streamlit 前端
|-- llm/                    检索、向量库、重排、模型接入
|-- test/                   图性能与评测脚本
|-- chroma_db/              本地 Chroma 数据目录
|-- project_config.py       全局配置入口
|-- requirements.txt        Python 依赖
|-- API_SPEC.md             接口说明
`-- test_main.http          HTTP 调试示例
```

## 技术栈

- Python
- FastAPI
- Streamlit
- SQLAlchemy
- LangChain
- LangGraph
- ChromaDB
- OpenAI API
- FlagEmbedding / Transformers / Torch

## 运行前准备

建议环境：

- Python 3.10+
- MySQL 8+
- 可用的 OpenAI API Key

安装依赖：

```bash
pip install -r requirements.txt
```

如果你本地没有安装某些扩展依赖，也可能需要补充：

```bash
pip install aiomysql pydantic[email] python-dotenv python-multipart
```

## 环境变量

项目配置集中在 [project_config.py](/d:/daima/agent/project_config.py)。

常用环境变量如下：

```ini
OPENAI_API_KEY=your_openai_key

LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

API_BASE_URL=http://127.0.0.1:8000/api
ASYNC_DATABASE_URL=mysql+aiomysql://root:123456@localhost:3306/agent?charset=utf8mb4

VECTOR_COLLECTION=knowledge_base
MD5_PATH=./md5.txt

TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50

UPLOAD_TIMEOUT_SECONDS=1800

RETRIEVAL_PROFILE=online
USE_QUERY_REWRITE=false
FINAL_RANK_ENABLED=true
RERANK_ENABLED=true
RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANK_DEVICE=cuda:0
```

说明：

- `OPENAI_API_KEY` 是模型调用必需项
- `ASYNC_DATABASE_URL` 用于用户、令牌、文档记录等关系型数据
- `API_BASE_URL` 供 Streamlit 前端调用后端接口
- `chroma_db/` 用于保存本地向量库数据

## 启动方式

### 1. 启动后端

项目根目录下执行：

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

默认可访问：

- 首页：`http://127.0.0.1:8000/`
- Swagger：`http://127.0.0.1:8000/docs`

说明：

- 后端启动时会执行数据库初始化
- 当前根入口 `main.py` 会暴露 FastAPI 应用

### 2. 启动前端

```bash
streamlit run web/app.py
```

前端包含两个主要工作区：

- 文档上传与向量库管理
- 旅行对话与智能问答

## 主要接口

当前已经接入的核心接口包括：

### 鉴权

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/profile`
- `POST /api/auth/change-password`

### 向量库

- `POST /api/vector-store/upload`
- `GET /api/vector-store/documents`

### 对话

- `POST /api/chat/completion`

更细的请求示例可以参考：

- [API_SPEC.md](/d:/daima/agent/API_SPEC.md)
- [test_main.http](/d:/daima/agent/test_main.http)

## 对话路由说明

核心图编排位于 [chat_graph.py](/d:/daima/agent/backend/app/graphs/chat_graph.py)。

系统会先对用户问题做预处理，再路由到不同能力链路：

- `ticket`
  适合车票、余票、车次、购票、改签等问题
- `roadmap`
  适合路线、导航、景点串联、游玩顺序等问题
- `rag`
  适合基于已上传文档的知识检索问答
- `other`
  适合预算、出行建议、场景性咨询等通用问题

此外，图中还包含：

- 上下文承接
- 问题改写
- 确认门控
- 结果校验
- 最终摘要整理

## 知识库与检索

知识库相关代码主要位于 `llm/` 目录。

当前实现包含：

- 文档加载与解析
- 文本切分
- 向量写入 Chroma
- 检索与重排
- 面向用户隔离的 collection 命名方式

上传文档后，后端会：

1. 解析文件内容
2. 补充元数据
3. 写入用户专属向量库
4. 在数据库中记录文件上传信息

## 测试与评估

`test/` 目录中提供了图编排的性能与效果评估脚本：

- `python test/run_graph_perf.py`
- `python test/run_graph_eval.py`

支持的典型能力：

- 并发压测
- 路由准确率评估
- `completed / needs_confirmation` 状态统计
- `verification.is_complete` 完整性评估
- 延迟与吞吐统计

评估结果会输出到 `test/results/`。

## 常见问题

### 1. 前端无法连接后端

优先检查：

- 后端是否已经启动
- `API_BASE_URL` 是否正确
- 默认地址是否为 `http://127.0.0.1:8000/api`

### 2. 模型调用失败

优先检查：

- `OPENAI_API_KEY` 是否已配置
- 当前模型名是否可用
- 网络是否能访问对应模型服务

### 3. 文档上传失败

优先检查：

- 文件格式是否受支持
- 解析时间是否超过 `UPLOAD_TIMEOUT_SECONDS`
- 向量库目录是否可写

### 4. 数据库初始化失败

优先检查：

- MySQL 是否启动
- `ASYNC_DATABASE_URL` 是否正确
- 数据库名称是否已经创建

## 开发说明

几个关键入口文件：

- [main.py](/d:/daima/agent/main.py)
- [backend/app/main.py](/d:/daima/agent/backend/app/main.py)
- [web/app.py](/d:/daima/agent/web/app.py)
- [project_config.py](/d:/daima/agent/project_config.py)

如果你准备继续扩展这个项目，通常会从下面几个方向切入：

- 增加新的路由类型或工具接入
- 优化 `chat_graph` 的分类与确认逻辑
- 丰富知识库解析格式
- 增加更完整的评测集和自动化测试

## 当前定位

这个项目很适合作为：

- 毕设/课程项目原型
- RAG + Agent + 工作流编排实践样例
- 旅行场景智能助手 Demo
- 后续继续扩展成多工具旅行规划平台的基础版本
