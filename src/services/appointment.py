from __future__ import annotations

import time
from typing import Any

from utils import now_iso


class AppointmentService:
    def __init__(self) -> None:
        pass

    def create_appointment(self, *, channel: str, dept: str, timeslot: str) -> dict[str, Any]:
        appt_id = f"APT-{int(time.time() * 1000) % 1000000}"
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

