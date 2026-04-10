# SmartStress 论文技术总结与 Repo 对齐方向

更新时间：2026-03-18（Asia/Shanghai）  
来源论文：`D:\NUS\BMI5101\SmartStress\main.tex`

## 1. 论文核心技术内容总结

## 1.1 总体框架

论文给出的 SmartStress 是三层闭环架构：
1. `PhysioSense`（生理感知）  
2. `MindCare`（语义推理与共情支持）  
3. `Meta-Reflective Orchestrator`（状态机编排 + HITL 安全闸）

目标是把“可穿戴感知”与“Agentic 推理”融合为可解释、可干预的闭环系统，而不是单纯监测或单纯聊天。

## 1.2 PhysioSense（DNN + Attention）

论文定义的检测链路：
1. ECG 700Hz 流 -> 20s 窗口，1s 步长
2. 提取 12 维 HRV/频域/几何特征
3. 按用户 baseline 做归一化（相对偏离）
4. 模型结构：  
   - 特征 token embedding（d=32）  
   - inter-feature self-attention  
   - MLP + sigmoid 输出 stress probability
5. 可解释性：SHAP（全局 + 局部）解释主导特征与方向性

## 1.3 MindCare（LLM + RAG）

论文定义的 MindCare 不只是回复对话，而是：
1. 从文本抽取语义 stressor
2. 通过向量检索拿到证据化心理知识（CBT / mindfulness）
3. 用结构化 prompt 生成共情、约束化、安全化回复
4. 把生理信息（压力概率与解释信息）注入语义推理

## 1.4 Meta-Reflective Orchestrator（闭环调度）

论文强调三点：
1. 状态驱动的动态编排（非线性循环，不是一次性流水线）
2. 多模态融合（生理概率 + 文本 stressor）
3. HITL 安全门（涉及外部动作需用户确认）

并给出“拒绝后反思再规划”的回路。

## 1.5 论文实验主线

1. WESAD 压力检测性能（Acc/Prec/Rec/F1）
2. Attention 消融（w/o attention vs ours）
3. RAG 消融（Control vs Experimental）
4. 检索/语义指标（TF-IDF + BERTScore）
5. SHAP 可解释性分析

## 2. 当前 Repo 与论文对齐现状

## 2.1 已对齐部分

1. 已有 LangGraph 状态机与循环路由  
2. 已有 MindCare + TaskRelief + HITL 机制  
3. 已有 RAG 检索链路（TiDB 向量存储）  
4. 已有 A/B 实验脚本、TF-IDF 与 BERTScore评估脚本  

## 2.2 未对齐关键差距

1. `PhysioSense` 仍是 HR 启发式占位逻辑，未实现论文 DNN+Attention 流程  
2. 尚未实现 SHAP 解释产物接入状态与对话  
3. MindCare prompt 尚未完整体现论文中的“生理上下文注入 + few-shot 约束模板”  
4. Orchestrator 仍偏 demo 级，缺“拒绝后反思再规划”的显式策略层  
5. 评测尚未形成“论文指标一键复现实验流水线”（WESAD训练/验证 + CounselChat消融统一报告）

## 3. 优化方向（让 Repo 符合论文内容）

## 3.1 方向 A：先补齐“论文最小可实现主链”

目标：从“demo 可跑”升级为“论文主方法可复现”。

1. 在 `SA-CoreGraph` 引入 PhysioSense 模型适配层：  
   - 输入：12维特征向量  
   - 输出：`current_stress_prob` + 可选解释字段占位  
2. 在 `SA-RAGData` 固化检索接口契约：  
   - top-k、来源元数据、失败降级一致化  
3. 在 `SA-APIRuntime` 增加可重复运行入口：  
   - 统一 session / checkpoint / eval 启动参数

## 3.2 方向 B：可解释性对齐（SHAP 接入）

目标：对齐论文“解释性证据”主张。

1. 增加 SHAP 推理脚本与 artifact 输出（global + instance）  
2. 在状态中新增解释字段（例如 `physio_explanations`）  
3. MindCare 消费解释字段，把“生理证据”写入提示词

## 3.3 方向 C：MindCare 提示词结构化升级

目标：对齐论文 prompt 设计四块结构（角色、生理注入、约束、示例）。

1. 拆分 `mind_care_node` 的策略函数  
2. 建立 prompt template 版本化  
3. 增加安全约束单测（自伤、诊断、药物建议等）

## 3.4 方向 D：实验复现流水线统一

目标：让论文结果可在 repo 中一键复现实验。

1. 新增 `experiments/pipelines/`：  
   - WESAD 分类训练/评估  
   - RAG 消融评估（TF-IDF + BERTScore）  
2. 统一输出 schema 与版本号  
3. 报告模板直接映射论文 RQ1/RQ2/RQ3

## 3.5 方向 E：面向真实部署的安全与鲁棒性

目标：覆盖论文讨论中的未来工作方向。

1. 多模态扩展接口（EDA/PPG/活动与睡眠）  
2. 线上容错与可观测性（日志、trace、异常分层）  
3. HITL 策略细化（确认超时、拒绝后反思重规划）

## 4. 建议执行顺序（当前迭代）

1. 先做方向 A（主链对齐）  
2. 再做方向 B + C（解释与提示词）  
3. 然后做方向 D（实验复现）  
4. 最后推进方向 E（部署级增强）

---

如果按模块 owner 分派：  
- `SA-CoreGraph`：PhysioSense 适配层、状态字段、编排策略  
- `SA-RAGData`：检索契约、TiDB 性能与质量  
- `SA-APIRuntime`：统一运行入口与 API 稳定性  
- `SA-EvalOps`：论文 RQ 对应评测流水线与报告自动化
