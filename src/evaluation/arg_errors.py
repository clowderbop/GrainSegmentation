"""Shared argparse / CLI helpers for evaluation scripts."""

from __future__ import annotations

import argparse


def raise_cli_argument_error(
    message: str,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    if parser is None:
        raise ValueError(message)
    parser.error(message)
