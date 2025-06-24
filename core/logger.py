import json
import logging
import logging.config
from pathlib import Path


def setup_logging(config_path: str = "config/logging.json",
                  default_level: int = logging.INFO) -> None:
    """Setup logging configuration from JSON file."""
    path = Path(config_path)
    if path.is_file():
        with open(path, "r") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        )

