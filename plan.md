# 财报 PDF RAG 计划

## 项目目标

做一个面向财报 PDF 的最小 RAG 助手，先完成：

* PDF 解析与切块
* 向量化与检索
* 基于检索结果回答问题
* 返回页码引用

后续再逐步加：

* query rewrite
* 二次检索
* verifier
* LangGraph
* FastAPI

## 当前边界

现在做的：

* 单文档财报问答
* 本地 JSON 存 chunks 和 embeddings
* OpenRouter embedding + chat
* 命令行问答

现在不做的：

* 多 agent
* 微调
* 前端
* 复杂数据库
* 复杂权限系统

## 当前进度

### 已完成

* 已接入财报 PDF [`茅台24年年度报告.pdf`](/Users/peteryao/projects/CaibaoAgent/茅台24年年度报告.pdf)
* 已完成 [`ingest.py`](/Users/peteryao/projects/CaibaoAgent/ingest.py)
  * PDF 提取文本
  * 小块切分
  * 生成 [`chunks.json`](/Users/peteryao/projects/CaibaoAgent/data/processed/chunks.json)
* 已完成 [`retriever.py`](/Users/peteryao/projects/CaibaoAgent/retriever.py)
  * 生成 embeddings
  * 保存 [`embeddings.json`](/Users/peteryao/projects/CaibaoAgent/data/processed/embeddings.json)
  * 基于余弦相似度检索 top-k
* 已完成 [`agent.py`](/Users/peteryao/projects/CaibaoAgent/agent.py)
  * 调用 retriever 检索
  * 调用 OpenRouter chat 模型回答
  * 输出答案与引用页码
  * 支持 CLI 参数
* 已完成 `.env` 配置加载
* 已完成基础测试
  * ingest
  * retriever
  * agent

### 未完成

* 固定评测问题集
* query rewrite
* 二次检索
* verifier
* FastAPI 接口
* README
* demo 展示材料
* LangGraph 重构

## 第一阶段结论

第一周最小闭环已经基本完成。

当前已经可以做到：

`PDF -> chunks -> embeddings -> retrieve -> answer -> citations`

并且可以直接通过命令行提问，例如：

```bash
uv run python /Users/peteryao/projects/CaibaoAgent/agent.py "贵州茅台2024年的营业总收入是多少？"
```

## 下一步

建议按这个顺序继续：

1. 补 10 到 20 个固定测试问题
2. 做一个最简单的评测脚本
3. 加 query rewrite
4. 加二次检索
5. 加 verifier
6. 最后再补 FastAPI

## 当前文件

```text
CaibaoAgent/
├── ingest.py
├── retriever.py
├── agent.py
├── data/processed/chunks.json
├── data/processed/embeddings.json
└── tests/
```
