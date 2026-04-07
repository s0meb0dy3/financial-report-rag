# 财报 PDF RAG 计划

## 项目目标

做一个面向财报 PDF 的最小 RAG 助手，并在最小闭环跑通后，逐步把效果和工程化补起来。

当前主线目标：

* PDF 解析与切块
* 向量化与检索
* 基于检索结果回答问题
* 返回页码引用
* 固定题集评测

后续增强目标：

* query rewrite
* 二次检索 / rerank
* verifier
* FastAPI 接口
* LangGraph 重构

## 当前边界

现在做的：

* 单文档财报问答
* 本地 JSON 存 chunks 和 embeddings
* OpenRouter embedding + chat
* 命令行问答
* 固定问题集评测

现在不做的：

* 多 agent
* 微调
* 前端
* 复杂数据库
* 复杂权限系统
* 生产级部署

## 当前进度

### 已完成

* 已接入财报 PDF [`茅台2024年年度报告完整版.pdf`](/Users/peteryao/projects/CaibaoAgent/茅台2024年年度报告完整版.pdf)
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
* 已完成 [`eval.py`](/Users/peteryao/projects/CaibaoAgent/eval.py)
  * 加载固定问题集
  * 调用 agent 跑问答
  * 用模型 judge 评估答案
  * 输出评测结果 JSON
* 已完成固定评测问题集 [`questions.json`](/Users/peteryao/projects/CaibaoAgent/data/eval/questions.json)
* 已完成 [`README.md`](/Users/peteryao/projects/CaibaoAgent/README.md)
* 已完成 `.env` 配置加载
* 已完成基础测试
  * ingest
  * retriever
  * agent
  * eval

### 未完成

* 扩充固定评测问题集数量与覆盖面
* 细化评测维度
  * 区分检索失败、回答错误、引用页码不准
* query rewrite
* 二次检索 / rerank
* verifier
* FastAPI 接口
* demo 展示材料
* LangGraph 重构

## 第一阶段结论

第一阶段的最小闭环已经完成。

当前已经可以做到：

`PDF -> chunks -> embeddings -> retrieve -> answer -> citations -> eval`

并且已经具备：

* 命令行提问能力
* 固定问题集评测能力
* 基础单元测试
* 基础使用文档

## 当前判断

现阶段最重要的事，不是继续加框架，而是先把评测做扎实，再根据评测结果优化检索链路。

原因：

* 现在已经有最小可用系统
* 但还缺少足够稳定的效果反馈闭环
* 如果没有更细的评测，后面加 query rewrite / rerank / verifier 很难判断是否真的提升
* FastAPI 和 LangGraph 更偏工程包装，应该放在效果验证之后

## 最高优先级计划

未来计划很多，但当前阶段最重要的只有下面 5 件事：

1. 建立可靠评测基线
   * 扩充固定题集到 20 到 50 题
   * 覆盖数字题、事实题、时间题、跨页题、同义改写题
   * 固定住一版 baseline 结果，后续所有优化都和它对比
2. 提升评测可解释性
   * 在评测结果里尽量区分检索失败、回答错误、引用页码不准
   * 保留失败样本，方便复盘
3. 优先优化检索，不先堆框架
   * 先做 query rewrite
   * 再做二次检索 / rerank
   * 每做一步都要跑评测确认是否真的提升
4. 增加可信度控制
   * 加 verifier
   * 强化“资料不足时回答我不知道”
   * 让答案、证据、页码之间尽量一致
5. 补最基本的工程稳健性
   * 让模型名、代理、文件路径这类常见配置错误有更清晰的报错
   * 补充更稳定的日志和调试信息
   * 保持 CLI 体验简单可复现

## 下一步

建议按这个顺序继续：

1. 扩充固定问题集到 20 到 50 题
2. 覆盖更多题型
   * 数字题
   * 基础事实题
   * 时间题
   * 人物 / 公司信息题
   * 跨页整合题
3. 增强评测脚本
   * 输出更清晰的通过率和分类型统计
   * 尽量区分召回问题和生成问题
4. 加 query rewrite
5. 加二次检索 / rerank
6. 加 verifier
7. 最后再补 FastAPI
8. 等链路稳定后再考虑 LangGraph 重构

如果时间有限，优先只做前 4 项，不要过早投入 FastAPI、LangGraph 或 multi-agent。

## 一个更具体的近期路线

### 第二阶段：把评测闭环做稳

目标：

* 能稳定复现当前效果
* 能看出系统主要错在哪
* 能用统一指标比较改动前后效果

交付物：

* 更完整的 [`questions.json`](/Users/peteryao/projects/CaibaoAgent/data/eval/questions.json)
* 更细的 [`eval.py`](/Users/peteryao/projects/CaibaoAgent/eval.py) 输出结果
* 一份最近一次评测结果文件

### 第三阶段：优化检索效果

目标：

* 提高召回质量
* 降低答非所问
* 提高页码引用可靠性

交付物：

* query rewrite 版本
* rerank / 二次检索版本
* verifier 版本
* 与基线版本的评测对比

### 第四阶段：补工程化能力

目标：

* 让系统更容易演示和复用
* 为后续扩展成服务做准备

交付物：

* FastAPI 接口
* demo 展示材料
* 视情况决定是否做 LangGraph 重构

## 当前文件

```text
CaibaoAgent/
├── ingest.py
├── retriever.py
├── agent.py
├── eval.py
├── README.md
├── data/
│   ├── eval/questions.json
│   └── processed/
│       ├── chunks.json
│       └── embeddings.json
└── tests/
    ├── test_ingest.py
    ├── test_retriever.py
    ├── test_agent.py
    └── test_eval.py
```
