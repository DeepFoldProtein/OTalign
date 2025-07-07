import logging


class TqdmHandler(logging.StreamHandler):
    def __init__(self) -> None:
        logging.StreamHandler.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        from tqdm.auto import tqdm

        msg = self.format(record)
        tqdm.write(msg)


def setup_logging() -> None:
    # Setup root logger
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            handler.close()
            root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[TqdmHandler()],
        force=True,
    )
