import logging
import sys
import datetime as dt

logger = logging.getLogger(__name__)


def init_base_logging(path: str = None):
    path = path if path else sys.stdout
    logging.basicConfig(
        level=logging.INFO,
        stream=path,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_crawling_tasks():
    pass


if __name__ == "__main__":
    init_base_logging()
    logger.info("started")
    started = dt.datetime.now()
    run_crawling_tasks()
    finished = dt.datetime.now()
    logger.info(f"done after: {(finished-started).total_seconds()} seconds")
