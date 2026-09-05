# Runs the reusable CACE data pipeline from raw GeoTab extraction through
# storage, filtering, quality checks, and data audits. Each step is executed
# separately so failures are easy to identify and the workflow can be reused
# manually now and scheduled later without changing the underlying scripts.

import argparse
import csv
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

PROCESSING_LOG_DIR = (
    PROJECT_ROOT
    / "metadata"
    / "processing_logs"
)

LOG_FILE = (
    LOG_DIR
    / "run_pipeline.log"
)

PROCESSING_LOG_FILE = (
    PROCESSING_LOG_DIR
    / "pipeline_runs.csv"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSING_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)


PIPELINE_STEPS = [
    {
        "name": "incremental_extraction",
        "script": (
            PROJECT_ROOT
            / "geotab_pipeline"
            / "extraction"
            / "extract_incremental_statusdata.py"
        ),
    },
    {
        "name": "s3_raw_upload",
        "script": (
            PROJECT_ROOT
            / "geotab_pipeline"
            / "storage"
            / "upload_raw_to_s3.py"
        ),
    },
    {
        "name": "signal_filtering",
        "script": (
            PROJECT_ROOT
            / "cace"
            / "preprocessing"
            / "signal_filter.py"
        ),
    },
    {
        "name": "data_quality_check",
        "script": (
            PROJECT_ROOT
            / "cace"
            / "preprocessing"
            / "data_quality_check.py"
        ),
    },
    {
        "name": "signal_inventory",
        "script": (
            PROJECT_ROOT
            / "cace"
            / "audit"
            / "signal_inventory.py"
        ),
    },
    {
        "name": "frequency_analysis",
        "script": (
            PROJECT_ROOT
            / "cace"
            / "audit"
            / "frequency_analysis.py"
        ),
    },
    {
        "name": "gap_analysis",
        "script": (
            PROJECT_ROOT
            / "cace"
            / "audit"
            / "gap_analysis.py"
        ),
    },
    {
        "name": "vehicle_coverage",
        "script": (
            PROJECT_ROOT
            / "cace"
            / "audit"
            / "vehicle_coverage.py"
        ),
    },
]

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the authorized CACE ingestion and data-audit workflow."
        )
    )

    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate pipeline script paths without executing the workflow.",
    )

    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute GeoTab extraction, S3 upload, and data-audit steps.",
    )

    return parser.parse_args()


def append_pipeline_run_log(
    run_id,
    started_at,
    finished_at,
    status,
    failed_step=None,
):
    file_exists = (
        PROCESSING_LOG_FILE.exists()
    )

    duration_sec = (
        finished_at
        - started_at
    ).total_seconds()

    row = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_sec": round(
            duration_sec,
            2,
        ),
        "status": status,
        "failed_step": failed_step or "",
    }

    with PROCESSING_LOG_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_id",
                "started_at_utc",
                "finished_at_utc",
                "duration_sec",
                "status",
                "failed_step",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def validate_pipeline_steps():
    missing_scripts = [
        step["script"]
        for step in PIPELINE_STEPS
        if not step["script"].exists()
    ]

    if missing_scripts:
        missing = "\n".join(
            str(path)
            for path in missing_scripts
        )

        raise FileNotFoundError(
            "Pipeline script(s) not found:\n"
            f"{missing}"
        )


def run_step(step):
    step_name = step["name"]
    script = step["script"]

    logger.info(
        "Starting step: %s",
        step_name,
    )

    started_at = datetime.now(
        timezone.utc
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    finished_at = datetime.now(
        timezone.utc
    )

    duration_sec = (
        finished_at
        - started_at
    ).total_seconds()

    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(
                "%s | %s",
                step_name,
                line,
            )

    if result.stderr:
        for line in result.stderr.splitlines():
            if result.returncode == 0:
                logger.info(
                    "%s | %s",
                    step_name,
                    line,
                )
            else:
                logger.error(
                    "%s | %s",
                    step_name,
                    line,
                )

    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed "
            f"with exit code {result.returncode}"
        )

    logger.info(
        "Completed step: %s | %.2f sec",
        step_name,
        duration_sec,
    )


def main():
    args = parse_args()
    validate_pipeline_steps()

    if args.validate_only:
        logger.info("Pipeline validation passed. No steps were executed.")
        return

    started_at = datetime.now(
        timezone.utc
    )

    run_id = started_at.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    logger.info(
        "Starting CACE pipeline | run_id=%s",
        run_id,
    )

    failed_step = None

    try:
        for step in PIPELINE_STEPS:
            failed_step = step["name"]
            run_step(step)

    except Exception:
        finished_at = datetime.now(
            timezone.utc
        )

        append_pipeline_run_log(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            status="failed",
            failed_step=failed_step,
        )

        logger.exception(
            "CACE pipeline failed | "
            "run_id=%s | step=%s",
            run_id,
            failed_step,
        )

        raise

    finished_at = datetime.now(
        timezone.utc
    )

    append_pipeline_run_log(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        status="success",
    )

    duration_sec = (
        finished_at
        - started_at
    ).total_seconds()

    logger.info(
        "CACE pipeline complete | "
        "run_id=%s | duration=%.2f sec",
        run_id,
        duration_sec,
    )


if __name__ == "__main__":
    main()