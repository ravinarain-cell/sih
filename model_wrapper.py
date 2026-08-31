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
except Exception as exc:  # pragma: no cover
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


def _payload_to_feature_frame(payload: dict[str, Any]) -> pd.DataFrame:
    subjects = _as_list(payload.get("subjects", []))
    batches = _as_list(payload.get("batches", []))
    rooms = _as_list(payload.get("rooms", []))

    rows: list[dict[str, int]] = []
    for idx, subject in enumerate(subjects):
        subject_data = subject if isinstance(subject, dict) else {}
        batch = _pick_item(batches, idx)
        room = _pick_item(rooms, idx)

        day_of_week = _to_int(
            subject_data.get("day_of_week")
            if subject_data.get("day_of_week") is not None
            else batch.get("day_of_week")
            if isinstance(batch, dict) and batch.get("day_of_week") is not None
            else room.get("day_of_week")
            if isinstance(room, dict) and room.get("day_of_week") is not None
            else 1,
            1,
        )
        time_of_day = _to_int(
            subject_data.get("time_of_day")
            if subject_data.get("time_of_day") is not None
            else batch.get("time_of_day")
            if isinstance(batch, dict) and batch.get("time_of_day") is not None
            else room.get("time_of_day")
            if isinstance(room, dict) and room.get("time_of_day") is not None
            else 9,
            9,
        )
        weather_condition = _to_int(
            subject_data.get("weather_condition")
            if subject_data.get("weather_condition") is not None
            else batch.get("weather_condition")
            if isinstance(batch, dict) and batch.get("weather_condition") is not None
            else room.get("weather_condition")
            if isinstance(room, dict) and room.get("weather_condition") is not None
            else 1,
            1,
        )
        enrolled_students = _to_int(
            subject_data.get("enrolled_students")
            if subject_data.get("enrolled_students") is not None
            else batch.get("size")
            if isinstance(batch, dict) and batch.get("size") is not None
            else room.get("capacity")
            if isinstance(room, dict) and room.get("capacity") is not None
            else 40,
            40,
        )

        rows.append(
            {
                "day_of_week": day_of_week,
                "time_of_day": time_of_day,
                "weather_condition": weather_condition,
                "enrolled_students": enrolled_students,
            }
        )

    frame = pd.DataFrame(rows, columns=MODEL_COLUMNS)
    if trained_model is not None and hasattr(trained_model, "feature_names_in_"):
        expected = list(trained_model.feature_names_in_)
        frame = frame.reindex(columns=expected, fill_value=0)
    return frame


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

    feature_frame = _payload_to_feature_frame(data)
    predictions = trained_model.predict(feature_frame)

    timetable: list[dict[str, Any]] = []
    for idx, subject in enumerate(subjects):
        subject_data = subject if isinstance(subject, dict) else {}
        teacher = _pick_item(teachers, idx)
        room = _pick_item(rooms, idx)
        batch = _pick_item(batches, idx)

        prediction_value = float(predictions[idx]) if idx < len(predictions) else 0.0
        slot_day = (int(prediction_value) % 5) + 1
        slot_hour = 9 + (int(round(prediction_value)) % 5)
        slot = f"Day {slot_day} {slot_hour:02d}:00"

        timetable.append(
            {
                "subject_id": subject_data.get("subject_id") or f"S{idx + 1}",
                "teacher_id": teacher.get("teacher_id") or f"T{idx + 1}",
                "room_id": room.get("room_id") or f"R{idx + 1}",
                "batch_id": batch.get("batch_id") or f"B{idx + 1}",
                "slot": slot,
            }
        )

    return timetable
