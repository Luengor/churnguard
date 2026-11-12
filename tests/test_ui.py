from pathlib import Path

from streamlit.testing.v1 import AppTest

import ui

UI_SCRIPT = Path(__file__).resolve().parent.parent / "src" / "ui.py"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fake_post(path, json=None, timeout=None):
    if path.endswith("/explain"):
        n = len(ui.meta.feature_columns)
        return FakeResponse(
            {
                "shap_values": [0.0] * n,
                "base_value": 0.0,
                "data": [0.0] * n,
                "feature_names": ui.meta.feature_columns,
            }
        )
    return FakeResponse({"churn_probability": 0.5, "churn": False})


def test_form_submits_prediction_and_explanation(monkeypatch):
    monkeypatch.setenv("CHURNGUARD_API_URL", "http://api:8000")
    monkeypatch.setattr(ui.requests, "post", _fake_post)

    app = AppTest.from_file(str(UI_SCRIPT))
    app.run()

    assert not app.exception
    app.button[0].click().run()

    assert not app.exception
    assert app.metric[0].value == "50.0%"
