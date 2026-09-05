# Provides a small reusable helper for recording CACE pipeline processing history.
# Each pipeline step can append one row describing what was processed, how many
# records were handled, and whether the step completed successfully.

import csv
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = (
    PROJECT_ROOT
    / "metadata"
    / "processing_logs"
)

LOG_FILE = (
    LOG_DIR
    / "processing_log.csv"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FIELDNAMES = [
    "processed_at_utc",
    "pipeline_step",
    "vehicle",
    "source_file",
    "output_file",
    "input_records",
    "output_records",
    "status",
    "message",
]


def append_processing_log(
    pipeline_step,
    status,
    vehicle=None,
    source_file=None,
    output_file=None,
    input_records=None,
    output_records=None,
    message=None,
):
    row = {
        "processed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pipeline_step": pipeline_step,
        "vehicle": vehicle,
        "source_file": (
            Path(source_file).name
            if source_file
            else None
        ),
        "output_file": (
            Path(output_file).name
            if output_file
            else None
        ),
        "input_records": input_records,
        "output_records": output_records,
        "status": status,
        "message": message,
    }

    file_exists = LOG_FILE.exists()

    with LOG_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)