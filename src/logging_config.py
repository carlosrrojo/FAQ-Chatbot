import logging
import sys

def configure_logging(level=logging.INFO):
    """
    Configures the root logger with a consistent format and level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Silence third-party noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
