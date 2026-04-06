from datetime import datetime
from zoneinfo import ZoneInfo


def get_current_run_datetime_str() -> str:
    timezone = ZoneInfo("Asia/Jakarta")
    run_datetime = datetime.now(tz=timezone).strftime("%Y-%m-%d_%H-%M-%S")

    return run_datetime


def get_current_run_datetime() -> datetime:
    timezone = ZoneInfo("Asia/Jakarta")
    run_datetime = datetime.now(tz=timezone)

    return run_datetime
