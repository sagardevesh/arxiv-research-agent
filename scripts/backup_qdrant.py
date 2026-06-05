#!/usr/bin/env python3
"""Snapshot Qdrant and upload to S3 for disaster recovery.

Creates a Qdrant collection snapshot, downloads it locally, uploads it to S3,
then deletes the local file. Run this on a cron schedule on the EC2 instance
(e.g. daily via crontab) to keep paper data safe across instance replacements.

Usage:
    python scripts/backup_qdrant.py

Required env vars:
    S3_BUCKET       S3 bucket name
    AWS_REGION      AWS region (default: us-east-1)

Optional env vars:
    QDRANT_URL      Qdrant base URL (default: http://localhost:6333)
    COLLECTION_NAME Qdrant collection (default: arxiv_papers)
"""

import os
import sys
import time
from pathlib import Path

import boto3
import httpx
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "arxiv_papers")
S3_BUCKET = os.getenv("S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def main() -> None:
    if not S3_BUCKET:
        print("ERROR: S3_BUCKET env var is not set.", file=sys.stderr)
        sys.exit(1)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(f"Creating Qdrant snapshot for collection '{COLLECTION}'...")
    resp = httpx.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/snapshots",
        timeout=120,
    )
    resp.raise_for_status()
    snapshot_name = resp.json()["result"]["name"]
    print(f"  snapshot: {snapshot_name}")

    local_path = Path(f"/tmp/{snapshot_name}")
    print("Downloading snapshot...")
    with httpx.stream(
        "GET",
        f"{QDRANT_URL}/collections/{COLLECTION}/snapshots/{snapshot_name}",
        timeout=300,
    ) as r:
        r.raise_for_status()
        with open(local_path, "wb") as fh:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
    size_mb = local_path.stat().st_size / 1e6
    print(f"  downloaded {size_mb:.1f} MB")

    s3_key = f"qdrant-snapshots/{timestamp}/{snapshot_name}"
    print(f"Uploading to s3://{S3_BUCKET}/{s3_key} ...")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    local_path.unlink()

    print(f"Backup complete → s3://{S3_BUCKET}/{s3_key}")


if __name__ == "__main__":
    main()
