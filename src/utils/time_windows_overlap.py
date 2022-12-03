import datetime
from typing import Tuple, Optional


def time_windows_overlap(
        date_range_1: Tuple[datetime.date, datetime.date],
        date_range_2: Tuple[datetime.date, datetime.date]
) -> Optional[Tuple[datetime.date, datetime.date]]:
    latest_start = max(date_range_1[0], date_range_2[0])
    earliest_end = min(date_range_1[1], date_range_2[1])
    delta = (earliest_end - latest_start).days + 1
    overlap = max(0, delta)
    if overlap:
        return latest_start, earliest_end
