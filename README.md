# README

本项目的检索增强生成（RAG）流程如下：系统启动后先加载或生成向量库，随后读取模板题目并按profile配置检索参数；接着执行向量检索、句级重排与可选扩句，构建上下文；再渲染prompt并调用LLM生成结构化回答；最后把答案与上下文写入`answers_and_contexts.json`，同时输出同批次`performance.json`，`main.py`在终端提示两份成果文件路径。详情请见下文。

## 文档处理与向量化

### 数据读取（`src/utils.py::ReadFiles`）

遍历`AI_database`下所有`.txt/.md/.pdf`，并按需触发OCR（默认读取文字层，必要时调用RapidOCR）。

### 分块策略
`get_chunk`以token为粒度（默认600tokens，含150token重叠区域）切分长文本，确保每个块在嵌入长度限制内，同时保留跨块的语义连续性。每段内容都会添加【来源：xxx】前缀，为后续引用准备。

### 嵌入生成（`src/Embeddings.py::OpenAIEmbedding`）
使用OpenAIEmbeddingsAPI计算chunk级向量，内置指数退避与最小间隔节流，保证在高并发/限速环境下稳定重试。生成的向量与原文档列表分别持久化为`storage/vectors.json`与`document.json`，供启动时快速加载。

## 检索与上下文构造

### 问题类型自适应
优先读取模板配置内的`question_type`；若无，再由`classify_question`按关键词与标点推断「直接回答」「轻分析」「深分析」。`get_question_config`据此给出`top_k`、上下文token上限、扩句范围等参数。

### 配置覆写
模板中的profile（如`info_search`、`multi_doc_analysis`）与题目可提供额外`retrieval_parameters`，通过`apply_retrieval_overrides`做字段级兜底与合法性检查，确保不会突破全局上限。

### 初次召回
`VectorStore.query`直接使用向量余弦相似度返回`top_k`条chunk。检索耗时会纳入性能日志。

### 语义扩展（`expand_context_with_source`）
根据问题提取关键词，回源文件查找高分句子并追加「【来源扩展】」块，从而补充细粒度背景。

### 句级重排（`build_limited_context`）
将候选块拆成句子，结合关键词命中得分，分阶段选择（优先取每块的种子句，再按分值全局排序），并跟踪token消耗以严格控制`max_tokens`。这一整合策略能在多文档、多段落情况下提供更高密度的证据片段。

### 自适应降采样
若LLM调用抛出`BadRequestError`（通常因token超限），会将`context_limit`递减并重试，最多3次，保证任务可恢复。

## 生成与提示词设计

### 提示模版体系
每个profile可指定`prompt_template`与LLM参数（如温度）；若题目未自定义，则回退到默认的`RAG_PROMPT_TEMPLATE`（`src/LLM.py`），该模板强调：角色扮演、证据链推理、引用格式、确定性分级等约束，确保输出结构化且可追溯。

### 模型调用（`src/LLM.py::OpenAIChat`）
通过CLOUD API接入大模型，实时打印流式输出方便调试。真正发送的消息为单轮userprompt（由模板渲染`question`与聚合`context`）。

### 答案结构
`src/rag_pipeline.py::answer_questions`会为每道题记录`question`、原始召回上下文（未截断）和模型回答，并保存到`output/answers_and_contexts.json`。
运行性能与Token估算

### 计时策略
`time.perf_counter()`分别量测召回与生成阶段，精确到毫秒，覆盖多次LLM重试的总时长。

### Token估算（`src/token_estimator.py`）
`estimate_tokens`用“4字符 = 1 token”估计token数。

### 性能日志
每道题都会记录`retrieval_ms`、`generation_ms`、`prompt_tokens_est`、`answer_tokens_est`，结构化写入`output/performance.json`。若后续需要监控平均耗时或对齐成本，可在该文件基础上追加聚合逻辑。