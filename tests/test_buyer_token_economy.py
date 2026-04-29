from __future__ import annotations

import json

from src.pipeline.buyer.token_economy import BuyerTokenEconomy
from src.pipeline.buyer.token_store import BuyerTokenStore


def test_buyer_token_economy_applies_cost_and_persists(tmp_path):
    store = BuyerTokenStore(data_dir=tmp_path / "buyer_data", default_total=100)
    economy = BuyerTokenEconomy(store=store)

    decision = economy.evaluate_request(session_id="session-a")

    assert decision["allowed"] is True
    assert decision["used_this_request"] == 29
    assert decision["total_used"] == 29
    assert decision["remaining"] == 71

    state = json.loads((tmp_path / "buyer_data" / "budget_state.json").read_text(encoding="utf-8"))
    assert state["sessions"]["session-a"]["used"] == 29
    assert state["sessions"]["session-a"]["remaining"] == 71


def test_buyer_token_economy_blocks_when_budget_insufficient(tmp_path):
    store = BuyerTokenStore(data_dir=tmp_path / "buyer_data", default_total=100)
    economy = BuyerTokenEconomy(store=store)

    # 3 successful requests consume 87; 4th request (29) must be blocked.
    assert economy.evaluate_request(session_id="session-b")["allowed"] is True
    assert economy.evaluate_request(session_id="session-b")["allowed"] is True
    assert economy.evaluate_request(session_id="session-b")["allowed"] is True
    blocked = economy.evaluate_request(session_id="session-b")

    assert blocked["allowed"] is False
    assert blocked["used_this_request"] == 29
    assert blocked["remaining"] == 13

