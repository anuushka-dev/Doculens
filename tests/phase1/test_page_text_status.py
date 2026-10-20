"""Boundary tests for extracted-page text classification."""

import pytest

from app.schemas.document import ExtractedPage


@pytest.mark.parametrize(
    ("text", "expected_status"),
    [
        ("", "empty_text"),
        (" " * 25, "empty_text"),
        ("x" * 19, "near_empty_text"),
        ("x" * 20, "ok"),
        ("x" * 21, "ok"),
    ],
)
def test_page_text_status_threshold_boundaries(
    text: str,
    expected_status: str,
) -> None:
    page = ExtractedPage(page_number=1, text=text)

    assert page.status == expected_status
