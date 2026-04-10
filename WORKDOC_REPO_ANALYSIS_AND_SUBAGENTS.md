# SmartStress Repo 工作文档（现状分析 + Subagent 分工）

更新时间：2026-03-18（Asia/Shanghai）  
分析基线：`main` 分支当前 tracked files（`git ls-files`）

## 1. 仓库目前已经做了什么（总览）

### 1.1 主业务能力（LangGraph 多智能体）

- 已实现一个可循环的工作流图：`physio_sense -> mind_care -> (task_relief_propose | execute_tool | wait_for_human_input | end)`。
- 已实现三类核心节点：
  - `PhysioSense`：基于生理输入计算当前压力概率（当前是启发式占位版本）。
  - `MindCare`：对话、stressor 识别、RAG 检索接入、用户确认流程处理。
  - `TaskRelief`：给出低风险任务调整建议，用户确认后执行 mock tool。
- 关键证据：
  - `smartstress_langgraph/graph.py`
  - `smartstress_langgraph/nodes/physio_sense_node.py`
  - `smartstress_langgraph/nodes/mind_care_node.py`
  - `smartstress_langgraph/nodes/task_relief_nodes.py`

### 1.2 会话与持久化

- 已使用 `langgraph.checkpoint.sqlite.SqliteSaver` 做 checkpoint 持久化，数据库文件为 `smartstress.db`。
- 会话 thread_id 规则：`{user_id}:{session_id}`。
- `start/continue` 都通过同一个 `APP` 编译图执行，支持跨调用恢复状态。
- 关键证据：
  - `smartstress_langgraph/graph.py`
  - `smartstress_langgraph/api.py`
  - `verify_persistence.py`

### 1.3 API 与服务入口

- 已有高层 SDK API：
  - `start_monitoring_session()`
  - `continue_session()`
  - `ingest_documents()`
- 已有 FastAPI 服务：
  - `POST /api/start_session`
  - `POST /api/continue_session`
  - `GET /health`
- 支持静态前端托管（通过 `FRONTEND_PATH` 或默认 `frontend_dist`）。
- 关键证据：
  - `smartstress_langgraph/api.py`
  - `server.py`

### 1.4 RAG 与数据处理

- 已切换到 TiDB 向量存储路径（文档表 + embedding 表）。
- 已有文档加载、嵌入、写库、相似检索链路。
- 已有 CounselChat 的 CSV->Markdown 转换与分批入库脚本。
- 关键证据：
  - `smartstress_langgraph/rag/ingestion.py`
  - `smartstress_langgraph/rag/tidb_vector_store.py`
  - `smartstress_langgraph/rag/retrieval.py`
  - `convert_counselchat_to_md.py`
  - `ingest_counselchat_tidb.py`

### 1.5 评测与实验

- 已有 A/B 测试框架（Control vs Experimental use_rag）。
- 已有 TF-IDF 相似度评估、BERTScore 评估、报告生成（单指标和组合报告）。
- 已有测试查询集与历史结果文件夹（`experiments/report/`）。
- 关键证据：
  - `experiments/run_ab_test.py`
  - `experiments/evaluate_results.py`
  - `experiments/evaluate_bertscore.py`
  - `experiments/generate_report.py`
  - `experiments/generate_report_combined.py`

---

## 2. 端到端调用链（当前真实实现）

### 2.1 启动会话链路

1. `server.py` 的 `POST /api/start_session` 收到请求。
2. 反序列化到 `StartSessionRequest`。
3. 调用 `smartstress_langgraph.api.start_monitoring_session()`。
4. `start_monitoring_session()` 组装初始 state，构建 `thread_id=user_id:session_id`。
5. `APP.invoke(initial_state, config={thread_id})` 触发图执行。
6. 返回 `SessionHandleModel + SmartStressStateView`。

### 2.2 继续会话链路

1. `server.py` 的 `POST /api/continue_session` 收到请求。
2. 反序列化到 `ContinueSessionRequest`。
3. 调用 `smartstress_langgraph.api.continue_session()`。
4. 内部先用 `APP.get_state(config)` 拉取 checkpoint 状态（`_load_cached_state`）。
5. 合并新 sensor_data / user_message 到 state。
6. 再次 `APP.invoke(state, config)` 继续图执行。
7. 返回更新后的 `SessionHandleModel + SmartStressStateView`。

### 2.3 HITL（人类确认）闭环

1. `TaskRelief` 先生成 `suggested_action`。
2. `MindCare` 把建议转成确认问题，设置 `awaiting_human_confirmation=True`。
3. 用户回答后，`MindCare` 规范化为 `yes/no/cancel`。
4. 路由器 `route_after_mind_care()` 决定：
  - `yes` -> `execute_tool`
  - `no/cancel` -> `monitoring_loop`（回到 `physio_sense`）
  - 否则按规则继续。

---

## 3. 模块盘点（文件级）

### 3.1 CoreGraph 模块

范围：`smartstress_langgraph/graph.py`, `state.py`, `nodes/*.py`, `llm/prompts.py`

已实现：

- 状态结构：`SmartStressState` 覆盖压力轨迹、对话历史、RAG 上下文、确认流、审计和错误日志。
- 路由逻辑集中在 `route_after_mind_care()`，图结构清晰。
- 各节点返回 “增量 updates dict”，符合 LangGraph 状态更新模式。

缺口：

- `physio_sense_node` 仍是 heuristic placeholder，未接真实模型。
- `mind_care_node` 逻辑分支较多且较长，维护成本高。
- 高压阈值 `>0.9` 写死，缺配置化。
- `graph.py` 有重复 import（`StateGraph, END` 重复）。

### 3.2 RAG/Data 模块

范围：`smartstress_langgraph/rag/*.py`, `convert_counselchat_to_md.py`, `ingest_counselchat_tidb.py`, `RAG_guide.md`

已实现：

- 文档加载支持 `.md/.txt`，统一映射到 `RagDocument`。
- TiDB 双表结构（documents + embeddings）可写入、可更新。
- `retrieve_context()` 失败时返回空列表，不阻断主流程。

缺口：

- `similarity_search()` 目前拉全量 embedding 到 Python 计算余弦相似度，扩展性弱。
- 每次检索都新建 store 连接并做表检查，运行开销大。
- 依赖声明不完整：`requirements.txt` 没显式列 `mysql-connector-python` 和 `python-dotenv`，却在代码中使用。
- 文档仍有 Chroma 叙述（`RAG_guide.md`），与 TiDB 实现不一致。

### 3.3 API/Runtime 模块

范围：`server.py`, `smartstress_langgraph/api.py`, `io_models.py`, `examples/*`, `run_api_key_test.py`, `test_api_conn.py`, `verify_persistence.py`, `smoke_test.py`

已实现：

- SDK + REST 双入口。
- 状态序列化为前端友好的 `SmartStressStateView`。
- 提供 smoke / API key / persistence 的脚本化验证。

缺口：

- `_load_cached_state()` 异常吞掉（`except Exception: pass`），故障排查不透明。
- `server.py` 默认 `allow_origins=["*"]` 仅适合开发，生产需收敛。
- README 提到 `test_memory_recall.py`，实际 tracked files 中不存在，文档偏差。

### 3.4 Eval/Experiment 模块

范围：`experiments/*.py`, `experiments/README.md`, `experiments/test_queries.json`

已实现：

- A/B runner 能驱动同一 query 在 use_rag=True/False 两组跑通。
- 评估支持：
  - TF-IDF 相似度（快速、确定性）
  - BERTScore（语义层）
- 报告支持统计、显著性检验、分类别分析。

缺口：

- `experiments/README.md` 写的是 LLM judge 流程，但 `evaluate_results.py` 已改为 TF-IDF only，存在说明不一致。
- `run_ab_test.py` 为切换 use_rag 使用 monkey patch `_load_cached_state`，可运行但可维护性一般。
- 若要实验可重复与可审计，建议固化随机种子、统一输出 schema 版本号。

---

## 4. 会话 / 记忆 / 持久化现状结论

1. 当前“记忆”本质是 `conversation_history + checkpoint state`，通过 SQLite 持久化可跨调用恢复。
2. session 唯一性依赖 `user_id:session_id`，如果外部重复使用同一键会复用状态。
3. 持久化验证脚本 `verify_persistence.py` 能从 DB 读取 `checkpoints` 表计数作为旁证。
4. 当前没有单独的“长期画像存储层”（如 profile DB）；`user_preferences` 仍在状态中传递。

---

## 5. 测试与验证脚本地图

- 连通性与凭证：`test_api_conn.py`, `run_api_key_test.py`
- 持久化：`verify_persistence.py`
- RAG 检索冒烟：`smoke_test.py`
- 示例流程：
  - `smartstress_langgraph/examples/demo_session.py`
  - `smartstress_langgraph/examples/ingest_docs_example.py`

当前观察：以“脚本验证”为主，缺标准化单元测试/集成测试目录（如 `tests/` + CI）。

---

## 6. 已识别问题清单（按优先级）

### P0（优先立即处理）

1. 文档与实现不一致：RAG 文档（Chroma 叙述）与实验说明（LLM judge 叙述）落后于代码。
2. 依赖声明不完整：缺 TiDB 连接相关包显式声明。
3. `physio_sense_node` 仍为占位算法，影响“智能检测”真实性。

### P1（应在下一迭代）

1. `mind_care_node` 复杂度高，需拆分策略函数并补测试。
2. RAG 检索从全量拉取改为数据库侧近邻检索（或预过滤）提升规模性能。
3. API 异常处理可观测性不足（吞异常、日志粒度不一致）。

### P2（持续优化）

1. FastAPI CORS/鉴权生产化配置模板。
2. 统一日志体系（替换零散 `print`）。
3. 引入 CI：最小 e2e（start->continue->confirm->execute）。

---

## 7. Subagent 组织设计（后续固定分工）

> 从本文件起，后续新任务默认按以下 owner 路由；跨模块任务由主责 subagent 牵头整合。


| Subagent        | 负责模块              | 文件边界（主）                                                                                                                                                                                                    | 核心职责                        | 不负责               |
| --------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | ----------------- |
| `SA-CoreGraph`  | Graph/State/Nodes | `smartstress_langgraph/graph.py`, `smartstress_langgraph/state.py`, `smartstress_langgraph/nodes/*`, `smartstress_langgraph/llm/prompts.py`                                                                | 状态机、路由、节点行为、安全确认流           | TiDB schema 与实验统计 |
| `SA-RAGData`    | RAG + 数据入库        | `smartstress_langgraph/rag/*`, `convert_counselchat_to_md.py`, `ingest_counselchat_tidb.py`, `RAG_guide.md`                                                                                                | 检索质量、入库流程、向量存储性能、数据清洗       | FastAPI/会话接口      |
| `SA-APIRuntime` | SDK/API/运行时       | `smartstress_langgraph/api.py`, `smartstress_langgraph/io_models.py`, `server.py`, `smartstress_langgraph/examples/*`, `run_api_key_test.py`, `test_api_conn.py`, `verify_persistence.py`, `smoke_test.py` | API 合约、会话生命周期、持久化稳定性、服务可运维性 | 评测指标设计            |
| `SA-EvalOps`    | 实验评测与报告           | `experiments/*`                                                                                                                                                                                            | A/B 流程、评估指标、统计显著性、报告模板      | 生产业务路由逻辑          |


### 7.1 当前会话的 owner 实例（可直接路由）


| 逻辑角色            | Agent 实例 ID                            | 昵称           |
| --------------- | -------------------------------------- | ------------ |
| `SA-CoreGraph`  | `019cffd1-57a0-78a3-a03d-f934e5073043` | `Ptolemy`    |
| `SA-RAGData`    | `019cffd1-6d1e-7df0-86d0-c57e112aa138` | `Pauli`      |
| `SA-APIRuntime` | `019cffd1-bc0a-73d1-91d5-25ec889fce98` | `Feynman`    |
| `SA-EvalOps`    | `019cffd1-d237-7102-a1fa-fb4ee8f6bf6b` | `Archimedes` |


---

## 8. 后续任务分派协议（执行规则）

### 8.1 单模块任务

- 直接派发给对应 owner subagent 处理并回传变更与验证结果。

### 8.2 跨模块任务

1. 先识别“改动最多/风险最高”的模块作为主责 subagent。
2. 其他 subagent 只提供接口层支持，不做主决策。
3. 合并前由主责 subagent 汇总冲突与联调结果。

### 8.3 冲突裁决

- 状态字段与图路由冲突 -> `SA-CoreGraph` 最终裁决。  
- TiDB schema / 检索策略冲突 -> `SA-RAGData` 最终裁决。  
- API contract / 服务行为冲突 -> `SA-APIRuntime` 最终裁决。  
- 指标定义与报告口径冲突 -> `SA-EvalOps` 最终裁决。

### 8.4 每次任务最小交付标准

1. 变更摘要（改了哪些文件、为什么）。
2. 验证证据（至少一个脚本/测试可复现）。
3. 风险说明（已知未覆盖点）。
4. 如改动 state/API，必须同步更新相关模型与文档。

---

## 9. 建议下一批任务（可直接派发）

1. 派给 `SA-RAGData`：统一 RAG 文档到 TiDB 版本，并补齐依赖声明。
2. 派给 `SA-CoreGraph`：拆分 `mind_care_node` 并提取可测试策略函数。
3. 派给 `SA-APIRuntime`：补一个最小 e2e 测试脚本链（start/continue/confirm）。
4. 派给 `SA-EvalOps`：修正实验 README 与当前评估脚本一致，补版本化报告头。

---

## 10. 分派关键字（供后续自动路由）

- 命中 `graph/state/node/route/HITL/stressor` -> `SA-CoreGraph`  
- 命中 `RAG/TiDB/embedding/retrieval/ingest/counselchat` -> `SA-RAGData`  
- 命中 `api/server/session/checkpoint/fastapi/persistence` -> `SA-APIRuntime`  
- 命中 `ab test/evaluate/bertscore/tfidf/report/metric` -> `SA-EvalOps`

> 若同时命中多个模块，按“文件改动范围最大”原则选择主责。  

---

## 11. 全程工作文档约定（2026-03-18 起生效）

1. `WORKDOC_REPO_ANALYSIS_AND_SUBAGENTS.md` 作为本 repo 的全程工作文档。  
2. 后续所有任务的目标、改动、验证和风险，均追加记录在此文件。  
3. 任何新增规划文档都需要在本文件留链接和执行状态。  

---

## 12. 本次执行记录（论文对齐阶段-第一步）

执行时间：2026-03-18（Asia/Shanghai）

### 12.1 已完成：P0 前两项修复

1. 修正 RAG 文档与实现不一致（Chroma -> TiDB）  
   - 文件：`RAG_guide.md`
   - 变更要点：
     - 改为 TiDB 向量存储说明
     - 增加 TiDB 环境变量配置示例
     - 更新重建索引方式为 TiDB 表清理

2. 修正实验文档与评估脚本不一致（LLM judge -> TF-IDF/BERTScore）  
   - 文件：`experiments/README.md`
   - 变更要点：
     - 主评估链路改为 `evaluate_results.py`（TF-IDF）
     - 补充 `evaluate_bertscore.py` 作为可选语义评估
     - 更新报告生成命令与文件说明

3. 补充主 README 的 RAG 与脚本说明一致性  
   - 文件：`README.md`
   - 变更要点：
     - RAG 描述从本地 `.rag_store` 改为 TiDB 表
     - 补充 TiDB 连接环境变量示例
     - 移除不存在的 `test_memory_recall.py` 引用

4. 修正依赖声明问题（补齐 TiDB 与实验运行所需）  
   - 文件：`requirements.txt`
   - 变更要点：
     - 新增 `mysql-connector-python`、`python-dotenv`
     - 新增 `google-genai`（匹配新 SDK 调用）
     - 新增实验脚本所需 `pandas`、`scikit-learn`、`scipy`
     - 移除已不再使用的 `chromadb`

### 12.2 已完成：论文技术总结与对齐方向文档

- 新增文档：`PAPER_MAIN_TEX_TECH_SUMMARY_AND_ALIGNMENT.md`  
- 来源：`D:\NUS\BMI5101\SmartStress\main.tex`  
- 内容：
  - 论文方法与实验主线总结
  - 当前 repo 与论文差距映射
  - 面向论文对齐的优化方向与执行顺序

### 12.3 当前状态

已完成你要求的第一阶段：  
- P0 前两项（文档与声明）已修正  
- 论文技术总结文档已落地  
- 给出了 repo 对齐论文的优化方向（见新文档第 3 节）

## 16. Overleaf Project Clone Log - 2026-03-24
- Source project URL: `https://www.overleaf.com/project/6935326d428d6b170bff5f48`
- Clone destination in current repo: `overleaf_project/`
- Clone method: Overleaf Git bridge using user-provided token (authenticated clone), then credential cleanup.
- Security cleanup applied:
  - Reset `origin` URL to token-free endpoint:
    - `https://git.overleaf.com/6935326d428d6b170bff5f48`
- Verification:
  - Files present: `main.tex`, `references.bib`, `main.pdf`, `ieeecolor.cls`, `generic.sty`, `logo.eps`, `figs/`.

## 17. Overleaf Paper Revision Pass - 2026-03-24
- Target project:
  - `D:\NUS\BMI5101\smart-stress-code\smart-stress-agent\overleaf_project`
- Authentication handling:
  - Added local-only `.env` in the Overleaf subrepo with project ID and git token.
  - Added `.gitignore` entry for `.env` so the token is not pushed to Overleaf.
- Main paper changes completed in `main.tex`:
  - Rewrote `Abstract`, `Introduction`, `Related Work`, `Methods`, `Discussion`, and `Conclusion`.
  - Converted the discussion into continuous prose.
  - Removed list-style contribution and discussion writing.
  - Updated figure captions to treat diagrams as authors' own illustrations.
  - Kept `MOODS` and `StressID` table placeholders.
  - Rewrote `RQ2` narrative while preserving the previous version as a local archive copy.
- Reference curation completed in `references.bib`:
  - Replaced the previous bibliography with a DOI-verified set of published references.
  - Current static count: 41 unique cited keys in `main.tex`.
  - Static check found no missing citation keys.
- Local archive material saved outside the Overleaf subrepo:
  - `D:\NUS\BMI5101\smart-stress-code\smart-stress-agent\paper_revision_archive\main_before_20260324.tex`
  - `D:\NUS\BMI5101\smart-stress-code\smart-stress-agent\paper_revision_archive\rq2_legacy_20260324.tex`
- Validation status:
  - Static LaTeX environment checks passed for figure, table, equation, and algorithm block pairing.
  - Local TeX compilation could not be run because `latexmk` and `pdflatex` are not installed in the current environment.
- Overleaf sync:
  - Committed in subrepo at `d551acb2a3813b54c522a6060996419450bc24f1`
  - Pushed successfully to remote `master`

## 18. SWELL Approx Mapping for DNN Evaluation - 2026-04-08
- Goal:
  - Build an approximate 12-feature mapping from SWELL HRV engineered CSVs for external evaluation against WESAD-trained DNN checkpoints.
- Implemented script:
  - `D:\NUS\BMI5101\smart-stress-model\evaluate_swell_approx_mapping.py`
- What the script does:
  - Maps SWELL engineered columns to the 12-feature order used by the current DNN.
  - Applies subject-level neutral median ratio normalization.
  - Exports mapped JSON datasets for testing with existing model code.
  - Supports fold-wise checkpoint evaluation using `Results_CrossVal_Full/cross_val_results.json` and `Models_CrossVal_Full`.
- Full mapping outputs generated:
  - `D:\NUS\BMI5101\smart-stress-model\Results_SWELL_Approx\SWELL_APPROX_TRAIN.json`
  - `D:\NUS\BMI5101\smart-stress-model\Results_SWELL_Approx\SWELL_APPROX_TEST.json`
- Generated dataset stats:
  - Train samples: 313310, feature dim: 12, labels: {1, 2}
  - Test samples: 39164, feature dim: 12, labels: {1, 2}
- Runtime note:
  - Current default Python environment cannot import PyTorch DLL correctly, so checkpoint inference could not be executed here.
  - Script supports `--skip-eval` for data generation-only mode and can run full evaluation in a torch-ready environment.
- Full evaluation run completed in conda env `torch`:
  - Command: `conda run -n torch python evaluate_swell_approx_mapping.py --output-dir Results_SWELL_Approx`
  - Fold-average metrics on SWELL mapped test set:
    - Accuracy: 0.4468
    - Precision: 0.4497
    - Recall: 0.9652
    - F1: 0.6132
  - Report file:
    - `D:\NUS\BMI5101\smart-stress-model\Results_SWELL_Approx\swell_approx_eval_report.json`
