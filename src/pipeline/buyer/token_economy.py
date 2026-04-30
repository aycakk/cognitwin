"""buyer/token_economy.py — fixed-cost budgeting for Buyer Agent only."""

from __future__ import annotations

from typing import Any

from src.pipeline.buyer.token_store import BuyerTokenStore, DEFAULT_BUYER_TOTAL


ACTION_COSTS: dict[str, int] = {
    "trend_analysis": 5,
    "sales_interpretation": 5,
    "assortment_planning": 7,
    "vendor_rfq": 4,
    "risk_analysis": 5,
    "final_recommendation": 3,
}

REQUEST_ACTION_SEQUENCE: tuple[str, ...] = (
    "trend_analysis",
    "sales_interpretation",
    "assortment_planning",
    "vendor_rfq",
    "risk_analysis",
    "final_recommendation",
)


class BuyerTokenEconomy:
    """Apply Buyer request token cost with a fixed action policy."""

    def __init__(self, store: BuyerTokenStore | None = None) -> None:
        self._store = store or BuyerTokenStore(default_total=DEFAULT_BUYER_TOTAL)

    def evaluate_request(self, session_id: str | None) -> dict[str, Any]:
        breakdown = {name: ACTION_COSTS[name] for name in REQUEST_ACTION_SEQUENCE}
        required = sum(breakdown.values())
        return self._store.consume(
            session_id=session_id,
            required_tokens=required,
            action_costs=breakdown,
        )

