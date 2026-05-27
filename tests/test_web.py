from anti_fraud_explorer.agent.models import TaskType
from anti_fraud_explorer.web.web import create_app


def test_meta_exposes_full_risk_level_enum():
    app = create_app()

    with app.test_client() as client:
        response = client.get("/api/meta")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["risk_levels"] == ["低", "中", "高", "极高", "无法判断"]


def test_ask_stream_returns_fallback_result_when_agent_crashes(monkeypatch):
    from anti_fraud_explorer.web import web as web_module

    def boom(*args, **kwargs):
        raise NameError("clamp_float")

    monkeypatch.setattr(web_module.Agent, "dispatch_stream", boom)

    app = create_app()
    app.testing = True

    with app.test_client() as client:
        response = client.post(
            "/api/ask", json={"question": "刷单诈骗是什么", "voice_enabled": False}
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["type"] == "result"
    assert payload["task_type"] == TaskType.FACT_QA.value
    assert "问答暂时失败" in payload["answer"]
