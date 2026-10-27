"""The browser's server (web_app.py): the dashboard must not fail on old or bad data."""

import json

import config
import web_app


def test_dashboard_survives_a_transaction_from_the_year_1000():
    # Recorded before dates were limited to 2000+; six months back from it is the year 999
    config.TRANSACTIONS_FILE.write_text(json.dumps([{
        "id": "t001", "type": "income", "date": "1000-02-05", "amount": 162.0, "category": "מתנות והטבות",
        "business": "שובר", "source_file": "x.png", "items": []}]), encoding="utf-8")
    result = web_app.dashboard("1000-02")
    assert result["summary"]["income_total"] == 162.0
    assert len(result["trend"]) == web_app.TREND_MONTHS
