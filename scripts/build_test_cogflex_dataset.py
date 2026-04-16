#!/usr/bin/env python3

from __future__ import annotations

from scripts.build_cogflex_dataset import (
    TEST_DATASET_ID,
    TEST_METADATA_PATH,
    TEST_QUALITY_REPORT_PATH,
    TEST_ROWS_PATH,
    build_test_artifacts,
    dataset_metadata,
    write_json,
)


def main() -> None:
    """Regenerate the tracked minimal test dataset artifacts."""
    rows, _answers, report = build_test_artifacts()
    write_json(TEST_ROWS_PATH, rows)
    write_json(TEST_QUALITY_REPORT_PATH, report)
    write_json(TEST_METADATA_PATH, dataset_metadata(TEST_DATASET_ID, "CogFlex Cognitive Flexibility Runtime Test"))


if __name__ == "__main__":
    main()
