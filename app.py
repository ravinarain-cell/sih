import importlib
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, ForeignKey, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

app = FastAPI(title="Timetable Generator API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./timetables.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)


class Base(DeclarativeBase):
    pass


class Timetable(Base):
    __tablename__ = "timetables"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entries: Mapped[list["TimetableEntry"]] = relationship(
        back_populates="timetable", cascade="all, delete-orphan"
    )


class TimetableEntry(Base):
    __tablename__ = "timetable_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    timetable_id: Mapped[int] = mapped_column(ForeignKey("timetables.id"), nullable=False)
    subject_id: Mapped[str | None] = mapped_column(String(255))
    subject: Mapped[str | None] = mapped_column(String(255))
    teacher_id: Mapped[str | None] = mapped_column(String(255))
    batch_id: Mapped[str | None] = mapped_column(String(255))
    room_id: Mapped[str | None] = mapped_column(String(255))
    slot: Mapped[str | None] = mapped_column(String(255))
    timetable: Mapped[Timetable] = relationship(back_populates="entries")


Base.metadata.create_all(engine)


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    errors = [
        {
            "field": ".".join(str(part) for part in error["loc"] if part != "body"),
            "constraint": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Invalid timetable payload", "errors": errors},
    )


class Teacher(BaseModel):
    teacher_id: str
    name: str
    subject: str | None = None
    availability: list[str] = Field(default_factory=list)


class Room(BaseModel):
    room_id: str
    name: str
    capacity: int = Field(..., gt=0)
    availability: list[str] = Field(default_factory=list)


class Subject(BaseModel):
    subject_id: str
    name: str
    teacher_id: str | None = None
    availability: list[str] = Field(default_factory=list)


class Batch(BaseModel):
    batch_id: str
    name: str
    size: int = Field(..., gt=0)
    availability: list[str] = Field(default_factory=list)


class TimetableRequest(BaseModel):
    teachers: list[Teacher]
    rooms: list[Room]
    subjects: list[Subject]
    batches: list[Batch]
    availability_constraints: dict[str, list[str]] = Field(default_factory=dict)


def _validate_scheduling_constraints(payload: TimetableRequest) -> None:
    errors: list[dict[str, str]] = []

    def check_unique(items: list[Any], key: str, resource: str) -> None:
        seen: set[str] = set()
        for item in items:
            value = getattr(item, key)
            if value in seen:
                errors.append({"field": f"{resource}.{key}", "constraint": f"duplicate id '{value}'"})
            seen.add(value)

    check_unique(payload.teachers, "teacher_id", "teachers")
    check_unique(payload.rooms, "room_id", "rooms")
    check_unique(payload.subjects, "subject_id", "subjects")
    check_unique(payload.batches, "batch_id", "batches")

    teachers = {teacher.teacher_id: teacher for teacher in payload.teachers}
    for subject in payload.subjects:
        if subject.teacher_id and subject.teacher_id not in teachers:
            errors.append({
                "field": f"subjects[{subject.subject_id}].teacher_id",
                "constraint": f"teacher '{subject.teacher_id}' does not exist",
            })

    default_slot_count = 4
    required_hours: dict[str, int] = {}
    for subject in payload.subjects:
        if subject.teacher_id:
            required_hours[subject.teacher_id] = required_hours.get(subject.teacher_id, 0) + len(payload.batches)
    for teacher_id, hours in required_hours.items():
        available_hours = len(teachers[teacher_id].availability) or default_slot_count
        if hours > available_hours:
            errors.append({
                "field": f"teachers[{teacher_id}].availability",
                "constraint": f"requires {hours} hours but only {available_hours} slots are available",
            })

    for batch in payload.batches:
        fitting_rooms = [room for room in payload.rooms if room.capacity >= batch.size]
        if not fitting_rooms:
            errors.append({
                "field": f"batches[{batch.batch_id}].size",
                "constraint": f"batch size {batch.size} exceeds every room capacity",
            })

    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Timetable constraints are invalid", "errors": errors},
        )


def _fallback_schedule(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Create a conflict-free schedule when no external scheduler is configured."""
    teachers = {item["teacher_id"]: item for item in data["teachers"]}
    subjects = data["subjects"]
    rooms = data["rooms"]
    constraints = data["availability_constraints"]
    default_slots = ["Monday 09:00", "Monday 10:00", "Tuesday 09:00", "Tuesday 10:00"]
    occupied: set[tuple[str, str]] = set()
    timetable: list[dict[str, Any]] = []

    batches = data["batches"]
    for subject in subjects:
        teacher_id = subject.get("teacher_id")
        teacher = teachers.get(teacher_id) if teacher_id else None
        if teacher_id and teacher is None:
            raise HTTPException(status_code=422, detail=f"Unknown teacher_id: {teacher_id}")

        for batch in batches:
            subject_slots = set(subject.get("availability", [])) or set(default_slots)
            if teacher and teacher.get("availability"):
                subject_slots &= set(teacher["availability"])
            if batch.get("availability"):
                subject_slots &= set(batch["availability"])
            if subject["subject_id"] in constraints:
                subject_slots &= set(constraints[subject["subject_id"]])
            slot = next(
                (candidate for candidate in subject_slots
                 if (not teacher_id or (teacher_id, candidate) not in occupied)
                 and any(
                     (item["room_id"], candidate) not in occupied
                     and (not item.get("availability") or candidate in item["availability"])
                     for item in rooms
                     if item["capacity"] >= batch["size"]
                 )),
                None,
            )
            if slot is None:
                raise HTTPException(status_code=409, detail="Unable to satisfy timetable constraints")
            room = next(
                item for item in rooms
                if item["capacity"] >= batch["size"]
                and (not item.get("availability") or slot in item["availability"])
                and (item["room_id"], slot) not in occupied
            )
            if teacher_id:
                occupied.add((teacher_id, slot))
            occupied.add((room["room_id"], slot))
            timetable.append({
                "subject_id": subject["subject_id"],
                "subject": subject["name"],
                "teacher_id": teacher_id,
                "batch_id": batch["batch_id"],
                "room_id": room["room_id"],
                "slot": slot,
            })
    return timetable


def _generate_with_model(data: dict[str, Any]) -> list[dict[str, Any]]:
    print("🚀 THE UPDATED MAIN.PY IS RUNNING!")
    
    # Hardcoded path to bypass environment variable issues
    model_path = "model_wrapper:generate_timetable"
    
    if not model_path:
        return _fallback_schedule(data)

    module_name, _, function_name = model_path.rpartition(":")
    if not module_name or not function_name:
        raise HTTPException(status_code=500, detail="SCHEDULING_MODEL must use module:function format")
    try:
        generator = getattr(importlib.import_module(module_name), function_name)
        result = generator(data)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Scheduling model failed: {exc}") from exc
    if not isinstance(result, list):
        raise HTTPException(status_code=500, detail="Scheduling model must return a list")
    return result


def _save_timetable(entries: list[dict[str, Any]]) -> int:
    with Session(engine) as session:
        timetable = Timetable(created_at=datetime.now(timezone.utc))
        timetable.entries = [
            TimetableEntry(
                subject_id=entry.get("subject_id"),
                subject=entry.get("subject"),
                teacher_id=entry.get("teacher_id"),
                batch_id=entry.get("batch_id"),
                room_id=entry.get("room_id"),
                slot=entry.get("slot"),
            )
            for entry in entries
        ]
        session.add(timetable)
        session.commit()
        session.refresh(timetable)
        return timetable.id


def _timetable_response(timetable: Timetable) -> dict[str, Any]:
    entries = [
        {
            "subject_id": entry.subject_id,
            "subject": entry.subject,
            "teacher_id": entry.teacher_id,
            "batch_id": entry.batch_id,
            "room_id": entry.room_id,
            "slot": entry.slot,
        }
        for entry in timetable.entries
    ]
    return {
        "status": "success",
        "timetable_id": timetable.id,
        "timetable": entries,
        "summary": {"entries": len(entries)},
    }


@app.get("/")
def read_root():
    return {"message": "Timetable API is running."}


@app.get("/ui")
def ui_page():
    return FileResponse("index.html")


@app.post("/generate-timetable")
def generate_timetable(payload: TimetableRequest):
    _validate_scheduling_constraints(payload)
    timetable = _generate_with_model(payload.model_dump())
    timetable_id = _save_timetable(timetable)
    return {
        "status": "success",
        "timetable_id": timetable_id,
        "timetable": timetable,
        "summary": {
            "teachers": len(payload.teachers),
            "rooms": len(payload.rooms),
            "subjects": len(payload.subjects),
            "batches": len(payload.batches),
            "entries": len(timetable),
        },
    }


@app.get("/timetable/{timetable_id}")
def get_timetable(timetable_id: int):
    with Session(engine) as session:
        timetable = session.scalar(
            select(Timetable).where(Timetable.id == timetable_id)
        )
        if timetable is None:
            raise HTTPException(status_code=404, detail="Timetable not found")
        return _timetable_response(timetable)