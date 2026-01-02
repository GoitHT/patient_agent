from __future__ import annotations

import random
from typing import Any

from utils import now_iso


class BillingService:
    def __init__(self, *, rng: random.Random) -> None:
        self.rng = rng

    def pay(self, *, order_id: str) -> dict[str, Any]:
        return {
            "order_id": order_id,
            "paid": True,
            "transaction_id": f"TX-{self.rng.randint(1000000, 9999999)}",
            "paid_at": now_iso(),
        }

