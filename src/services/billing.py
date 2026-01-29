from __future__ import annotations

import time
from typing import Any

from utils import now_iso


class BillingService:
    def __init__(self) -> None:
        pass

    def pay(self, *, order_id: str) -> dict[str, Any]:
        return {
            "order_id": order_id,
            "paid": True,
            "transaction_id": f"TX-{int(time.time() * 1000) % 10000000}",
            "paid_at": now_iso(),
        }

