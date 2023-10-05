"""This module contains the general configuration of the project."""
from __future__ import annotations

from pathlib import Path


SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()


__all__ = ["BLD", "SRC"]
