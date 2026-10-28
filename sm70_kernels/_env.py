"""Environment switches, kept independent of any inference engine."""

import logging
import os

logger = logging.getLogger(__name__)


def env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")
