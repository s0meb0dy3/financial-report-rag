# 系统架构

## 概览

当前项目是一个面向财报 PDF 的多文档 RAG 原型，核心链路如下：

`PDF -> chunks -> embeddings -> ChromaDB -> retrieve -> answer -> eval`

## 核心模块

### 1. 文档接入

[`ingest.py`](/Users/peteryao/projects/CaibaoAgent/ingest.py)

负责：

* 扫描多个 PDF
* 提取页面文本
* 按长度切块
* 为每个 chunk 添加 `doc_id`、`doc_name`、`source_path`、`page`

输出：

* [`chunks.json`](/Users/peteryao/projects/CaibaoAgent/data/processed/chunks.json)

### 2. 向量化与索引

[`retriever.py`](/Users/peteryao/projects/CaibaoAgent/retriever.py)
[`vector_store.py`](/Users/peteryao/projects/CaibaoAgent/vector_store.py)

负责：

* 调用 OpenRouter embedding 模型生成向量
* 将文本、向量和元数据写入 ChromaDB
* 基于 query embedding 检索 top-k chunk

存储：

* `data/chroma/`

### 3. 问答生成

[`agent.py`](/Users/peteryao/projects/CaibaoAgent/agent.py)

负责：

* 调用 retriever 检索相关 chunk
* 构造多文档上下文
* 调用 chat 模型生成答案
* 输出结构化 citations

当前引用格式：

* `doc_name + page`

### 4. 评测

[`eval.py`](/Users/peteryao/projects/CaibaoAgent/eval.py)
[`questions.json`](/Users/peteryao/projects/CaibaoAgent/data/eval/questions.json)

负责：

* 读取固定问题集
* 调用 Agent 跑问答
* 用模型 judge 评估结果
* 输出评测结果

## 当前特点

目前架构的特点是：

* 已支持多文档检索
* 已支持本地持久化向量库
* 已支持命令行问答
* 已支持基础评测

## 当前限制

目前还没有：

* query rewrite
* rerank / 二次检索
* verifier
* FastAPI 接口
* multi-agent 编排
