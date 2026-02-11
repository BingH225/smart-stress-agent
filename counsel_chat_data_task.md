这份文档为 Coding Agent 提供了详细的任务与计划，旨在利用 `counselchat_data.csv` 数据集增强 **SmartStress Agent** 的 RAG（检索增强生成）能力，并设计 A/B 测试实验来验证其对 **MindCare** 节点回复质量的提升。

---

# 任务文档：使用 CounselChat 数据集进行 RAG 增强与 A/B 测试实验

## 1. 任务目标

1. **数据转换**：将 `counselchat_data.csv` 格式的专家心理咨询数据转换为项目支持的 Markdown 格式。
2. **知识库集成**：将转换后的文档集成至 `rag_docs/` 并更新向量存储。
3. **实验设计 (A/B Test)**：设计并实现一套自动化实验框架，对比使用 CounselChat 增强前后的 MindCare 回复质量（准确性、共情度、忠实度）。

## 2. 执行计划 (Execution Plan)

### 第一阶段：数据预处理与转换 (CSV to Markdown)

* **任务描述**：编写解析脚本，处理 `counselchat_data.csv`。
* **具体步骤**：
1. 读取 CSV 文件，提取关键列（如 `questionText`, `answerText`, `topic`）。
2. 数据清洗：移除 HTML 标签、多余空格，并过滤掉内容过短的条目。
3. 格式化输出：为每一条问答生成一个独立的 `.md` 文件，存放在 `rag_docs/counselchat/` 目录下。
4. 元数据嵌入：在 Markdown 文件末尾添加来源标记 ``。



### 第二阶段：知识库构建与索引更新

* **任务描述**：将新生成的文档导入系统的向量存储。
* **具体步骤**：
1. 调用 `smartstress_langgraph.api.ingest_documents` 函数，指定文件夹路径为 `rag_docs/counselchat/`，并添加标签 `psychoeducation`。
2. 验证索引：确保 `smartstress_langgraph/.rag_store` 目录下成功生成或更新了索引文件。



### 第三阶段：A/B 测试实验设计 (Experiments)

* **任务描述**：构建对比实验环境，评估 RAG 增强的效果。
* **实验组设置**：
* **组 A (Control)**：使用原始的极简知识库或禁用 RAG 检索。
* **组 B (Experimental)**：启用 CounselChat 增强后的 RAG 系统。


* **指标设计**：
1. **Groundedness (忠实度)**：AI 回复内容是否严格基于 `rag_context`。
2. **Stressor Identification (压力源识别)**：评估 MindCare 从用户输入中提取 `current_stressor` 的准确率。
3. **Safety Compliance**：检查回复是否严格遵守不提供临床诊断的安全准则。



### 第四阶段：自动化执行与评估脚本编写

* **任务描述**：编写 Python 脚本批量运行测试用例。
* **具体步骤**：
1. 创建测试数据集：准备 20-50 个模拟高压力场景的用户输入。
2. 运行管道：利用 `start_monitoring_session` 和 `continue_session` 接口批量跑测。
3. 数据采集：捕获 `SmartStressStateView` 中的 `conversation_history`、`rag_context` 和 `audit_trail`。
4. 结果打分：利用 LLM-as-a-judge 机制或人工对组 A 和组 B 的回复进行盲评打分。



---

## 3. Coding Agent 关键代码模块参考

### CSV 转换逻辑片段

```python
import pandas as pd
from pathlib import Path

def convert_csv_to_md(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, row in df.iterrows():
        content = f"# Topic: {row['topic']}\n\n## Question\n{row['questionText']}\n\n## Expert Answer\n{row['answerText']}\n"
        with open(f"{output_dir}/cc_{i}.md", "w", encoding="utf-8") as f:
            f.write(content)

```

### 实验评估逻辑片段

```python
def run_ab_test(test_queries):
    # 组 A 和 组 B 的初始化与调用
    for query in test_queries:
        # 调用 API 获取回复
        handle, view = start_monitoring_session(req)
        # 记录 audit_trail 进行行为分析
        log_audit = view.audit_trail 

```

## 4. 验收标准

* [ ] `rag_docs/counselchat/` 目录下包含正确格式化的 Markdown 文件。
* [ ] 调用 `ingest_documents` 后返回的 `ingested` 数量与文件数一致。
* [ ] 实验报告能够展示组 B 在“回复专业度”和“建议具体性”上相对于组 A 的量化提升。