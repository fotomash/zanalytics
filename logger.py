import json
import logging
import logging.config
from pathlib import Path


def setup_logging(config_path: str = "config/logging.json", default_level: int = logging.INFO) -> None:
    """Setup logging configuration from JSON file."""
    path = Path(config_path)
    if path.is_file():
        with open(path, "r") as f:
            config = json.load(f)

        # Ensure directories for file handlers exist
        file_handler = config.get("handlers", {}).get("file", {})
        filename = file_handler.get("filename")
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        )

