import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_COLUMNS = ["day_of_week", "time_of_day", "weather_condition", "enrolled_students"]


def _resolve_model_path() -> Path:
    candidates = [
        Path(__file__).resolve().with_name("attendance_model.joblib"),
        Path.cwd() / "attendance_model.joblib",
        Path("attendance_model.joblib").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "attendance_model.joblib not found near the project root or script directory."
    )


try:
    MODEL_PATH = _resolve_model_path()
    trained_model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as exc:
    trained_model = None
    MODEL_PATH = None
    print(f"❌ Failed to load model: {exc}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick_item(items: list[Any], idx: int) -> dict[str, Any]:
    if not items:
        return {}
    item = items[idx % len(items)]
    return item if isinstance(item, dict) else {}


def generate_timetable(data: dict[str, Any] | str) -> list[dict[str, Any]]:
    if trained_model is None:
        raise ValueError("Model is not loaded; check attendance_model.joblib and installed dependencies.")

    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise TypeError("Input payload must be a dictionary or JSON string.")

    subjects = _as_list(data.get("subjects", []))
    teachers = _as_list(data.get("teachers", []))
    rooms = _as_list(data.get("rooms", []))
    batches = _as_list(data.get("batches", []))

    if not subjects:
        return []

    candidate_slots = [
        (day, hour)
        for day in range(1, 6)
        for hour in range(9, 17)
    ]

    occupied: set[tuple[str, str]] = set()
    timetable: list[dict[str, Any]] = []

    for idx, subject in enumerate(subjects):
        subject_data = subject if isinstance(subject, dict) else {}
        teacher = _pick_item(teachers, idx)
        room = _pick_item(rooms, idx)
        batch = _pick_item(batches, idx)

        teacher_id = teacher.get("teacher_id") or f"T{idx + 1}"
        room_id = room.get("room_id") or f"R{idx + 1}"
        batch_id = batch.get("batch_id") or f"B{idx + 1}"
        subject_id = subject_data.get("subject_id") or f"S{idx + 1}"

        enrolled_students = _to_int(
            subject_data.get("enrolled_students")
            or batch.get("size")
            if isinstance(batch, dict) else None
            or room.get("capacity")
            if isinstance(room, dict) else None,
            40,
        )
        weather_condition = _to_int(subject_data.get("weather_condition"), 1)

        rows = [
            {
                "day_of_week": day,
                "time_of_day": hour,
                "weather_condition": weather_condition,
                "enrolled_students": enrolled_students,
            }
            for day, hour in candidate_slots
        ]
        feature_frame = pd.DataFrame(rows, columns=MODEL_COLUMNS)

        if hasattr(trained_model, "feature_names_in_"):
            expected = list(trained_model.feature_names_in_)
            feature_frame = feature_frame.reindex(columns=expected, fill_value=0)

        predicted_scores = trained_model.predict(feature_frame)

        ranked_indices = sorted(
            range(len(predicted_scores)),
            key=lambda i: predicted_scores[i],
            reverse=True,
        )

        chosen_slot = None
        for slot_idx in ranked_indices:
            day, hour = candidate_slots[slot_idx]
            slot_name = f"Day {day} {hour:02d}:00"

            is_teacher_busy = (teacher_id, slot_name) in occupied
            is_room_busy = (room_id, slot_name) in occupied
            is_batch_busy = (batch_id, slot_name) in occupied

            if not is_teacher_busy and not is_room_busy and not is_batch_busy:
                chosen_slot = slot_name
                occupied.add((teacher_id, slot_name))
                occupied.add((room_id, slot_name))
                occupied.add((batch_id, slot_name))
                break

        if chosen_slot is None:
            day, hour = candidate_slots[ranked_indices[0]]
            chosen_slot = f"Day {day} {hour:02d}:00"

        timetable.append(
            {
                "subject_id": subject_id,
                "teacher_id": teacher_id,
                "room_id": room_id,
                "batch_id": batch_id,
                "slot": chosen_slot,
            }
        )

    return timetable