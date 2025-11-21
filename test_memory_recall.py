from __future__ import annotations
import json
from smartstress_langgraph.api import continue_session
from smartstress_langgraph.io_models import (
    ChatMessage,
    ContinueSessionRequest,
    SessionHandleModel,
)

def test_memory_recall():
    print("=== 开始测试：模拟重启后的记忆唤醒 ===")
    
    # 1. 手动构造与 run_api_key_test.py 完全一致的“凭证” (Handle)
    # 注意：这里必须和上一个脚本里的 user_id, session_id 一模一样
    reconstructed_handle = SessionHandleModel(
        user_id="demo-user",
        session_id="api-key-test",
        thread_id="demo-user:api-key-test",  # api.py 中的构造规则是 user_id:session_id
    )

    # 2. 直接“继续会话”，不调用 start_monitoring_session
    # 这会迫使系统去 SQLite 数据库里查找这个 thread_id 的历史
    recall_request = ContinueSessionRequest(
        session_handle=reconstructed_handle,
        user_message=ChatMessage(
            role="user",
            content="I just told you about a project demo. Do you remember what was stressing me out?",
        ),
    )

    print("正在发送新消息，等待 AI 读取数据库历史...")
    _, updated_state = continue_session(recall_request)

    # 3. 验证结果：打印 AI 的回复和完整的历史记录
    print("\n=== AI 的回复 (应该包含对 Slides/Demo 的提及) ===")
    # 取最后一条消息（AI 的新回复）
    if updated_state.conversation_history:
        last_msg = updated_state.conversation_history[-1]
        print(f"MindCare: {last_msg['content']}")
    
    print("\n=== 完整历史记录检查 (应包含上一轮对话) ===")
    print(f"历史消息总数: {len(updated_state.conversation_history)}")
    for i, msg in enumerate(updated_state.conversation_history):
        print(f"[{i}] {msg['role']}: {msg['content'][:50]}...")

if __name__ == "__main__":
    test_memory_recall()