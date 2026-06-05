"""Glossary-aligned merged-view test fixture module naming."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_merged_view_fixtures_module_is_importable() -> None:
    module = importlib.import_module("common.tests.merged_view_fixtures")
    assert callable(module.blank_map)
    assert callable(module.get_bundle_fixture)


def test_no_discouraged_instance_map_fixture_filename() -> None:
    tests_dir = Path(__file__).resolve().parent
    assert not (tests_dir / "instance_map_fixtures.py").is_file()
