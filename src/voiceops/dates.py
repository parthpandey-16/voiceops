"""Date arithmetic from KB-00.3: business days are Mon-Sat minus warehouse holidays."""
from __future__ import annotations

from datetime import date, timedelta

from .config import settings
from .kb import load_holidays


def today() -> date:
    return date.fromisoformat(settings.today)


def _parse(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d.isoformat() not in load_holidays()


def add_business_days(start: str | date, days: int) -> date:
    """The date `days` business days after `start` (the start day itself is day 0)."""
    d, left = _parse(start), int(days)
    while left > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            left -= 1
    return d


def add_calendar_days(start: str | date, days: int) -> date:
    return _parse(start) + timedelta(days=int(days))


def business_days_between(start: str | date, end: str | date) -> int:
    """Business days after `start` up to and including `end` (negative if end < start)."""
    a, b = _parse(start), _parse(end)
    sign = 1
    if b < a:
        a, b, sign = b, a, -1
    count, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if is_business_day(d):
            count += 1
    return sign * count


def calendar_days_between(start: str | date, end: str | date) -> int:
    return (_parse(end) - _parse(start)).days


def date_calc(operation: str, start: str, days: int | None = None, end: str | None = None) -> dict:
    """Tool entry point used by the agent."""
    start = start or settings.today
    if operation == "add_business_days":
        result = add_business_days(start, days or 0)
    elif operation == "add_calendar_days":
        result = add_calendar_days(start, days or 0)
    elif operation == "business_days_between":
        return {"business_days": business_days_between(start, end or settings.today),
                "note": "Mon-Sat, excluding holidays.csv"}
    elif operation == "calendar_days_between":
        return {"calendar_days": calendar_days_between(start, end or settings.today)}
    else:
        return {"error": f"unknown operation {operation}"}
    return {"date": result.isoformat(), "weekday": result.strftime("%A"),
            "is_past": result < today(), "today": settings.today}
