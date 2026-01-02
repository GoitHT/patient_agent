from __future__ import annotations

import random
from typing import Any

from utils import now_iso


class AppointmentService:
    def __init__(self, *, rng: random.Random) -> None:
        self.rng = rng

    def create_appointment(self, *, channel: str, dept: str, timeslot: str) -> dict[str, Any]:
        appt_id = f"APT-{self.rng.randint(100000, 999999)}"
        return {
            "appointment_id": appt_id,
            "channel": channel,
            "dept": dept,
            "timeslot": timeslot,
            "status": "booked",
            "created_at": now_iso(),
        }

    def checkin(self, appointment: dict[str, Any]) -> dict[str, Any]:
        appointment = dict(appointment)
        appointment.update({"status": "checked_in", "checked_in_at": now_iso()})
        appointment["waiting_status"] = "waiting"
        return appointment

    def call_patient(self, appointment: dict[str, Any]) -> dict[str, Any]:
        appointment = dict(appointment)
        appointment.update({"status": "called_in", "called_in_at": now_iso()})
        appointment["waiting_status"] = "in_room"
        return appointment

