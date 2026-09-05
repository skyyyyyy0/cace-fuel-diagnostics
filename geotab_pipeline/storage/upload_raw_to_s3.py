# Uploads locally collected GeoTab raw files to the CACE S3 raw data layer.
# Files are stored using the existing vehicle/year/month/day partition structure.
# Existing S3 objects are skipped so repeated pipeline runs do not create duplicates.

import logging
import os
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"
AWS_CONFIG = PROJECT_ROOT / "config" / "aws.yaml"

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "geotab"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

LOG_FILE = (
    LOG_DIR
    / "upload_raw_to_s3.log"
)

LOG_DIR.mkdir(
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


def load_aws_config():
    if not ENV_FILE.exists():
        raise FileNotFoundError(
            f".env file not found: {ENV_FILE}"
        )

    if not AWS_CONFIG.exists():
        raise FileNotFoundError(
            f"AWS config not found: {AWS_CONFIG}"
        )

    load_dotenv(ENV_FILE)

    with AWS_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file) or {}

    bucket_name = os.getenv(
        "CACE_S3_BUCKET"
    )

    region = os.getenv(
        "AWS_REGION"
    )

    raw_prefix = config.get(
        "raw_prefix",
        "raw/geotab",
    )

    if not bucket_name:
        raise ValueError(
            "CACE_S3_BUCKET is not defined in .env"
        )

    if not region:
        raise ValueError(
            "AWS_REGION is not defined in .env"
        )

    return {
        "bucket_name": bucket_name,
        "region": region,
        "raw_prefix": raw_prefix.strip("/"),
    }


def create_s3_client(region):
    return boto3.client(
        "s3",
        region_name=region,
    )


def find_raw_files():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}"
        )

    files = sorted(
        RAW_DIR.rglob("statusdata_*.csv")
    )

    logger.info(
        "Found %d raw file(s)",
        len(files),
    )

    return files


def build_s3_key(
    local_file,
    raw_prefix,
):
    relative_path = local_file.relative_to(
        RAW_DIR
    )

    return (
        f"{raw_prefix}/"
        f"{relative_path.as_posix()}"
    )


def object_exists(
    s3_client,
    bucket_name,
    s3_key,
):
    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key=s3_key,
        )

        return True

    except ClientError as exc:
        error_code = (
            exc.response
            .get("Error", {})
            .get("Code")
        )

        if error_code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False

        raise


def upload_file(
    s3_client,
    bucket_name,
    local_file,
    s3_key,
):
    s3_client.upload_file(
        str(local_file),
        bucket_name,
        s3_key,
    )


def main():
    logger.info(
        "Starting CACE raw S3 upload"
    )

    config = load_aws_config()

    s3_client = create_s3_client(
        config["region"]
    )

    raw_files = find_raw_files()

    if not raw_files:
        logger.info(
            "No raw files found for upload"
        )
        return

    uploaded = 0
    skipped = 0
    failed = 0

    for local_file in raw_files:
        s3_key = build_s3_key(
            local_file,
            config["raw_prefix"],
        )

        try:
            if object_exists(
                s3_client,
                config["bucket_name"],
                s3_key,
            ):
                logger.info(
                    "Skipping existing file: s3://%s/%s",
                    config["bucket_name"],
                    s3_key,
                )

                skipped += 1
                continue

            upload_file(
                s3_client,
                config["bucket_name"],
                local_file,
                s3_key,
            )

            logger.info(
                "Uploaded: %s -> s3://%s/%s",
                local_file.relative_to(
                    PROJECT_ROOT
                ),
                config["bucket_name"],
                s3_key,
            )

            uploaded += 1

        except Exception:
            logger.exception(
                "Failed to upload: %s",
                local_file,
            )

            failed += 1

    logger.info(
        "S3 upload complete | uploaded=%d | skipped=%d | failed=%d",
        uploaded,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()