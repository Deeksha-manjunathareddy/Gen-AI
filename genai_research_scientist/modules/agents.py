from __future__ import annotations

"""
Facade module that exposes all agent classes while delegating
their implementations to dedicated files.

This keeps the public import path stable:

    from modules.agents import SearchAgent, ExtractionAgent, ...

while still organizing each agent in its own file.
"""

from .search_agent import SearchAgent  # noqa: F401
from .extraction_agent import ExtractionAgent  # noqa: F401
from .hypothesis_agent import HypothesisAgent  # noqa: F401
from .experiment_agent import ExperimentAgent  # noqa: F401
from .draft_agent import DraftAgent  # noqa: F401



