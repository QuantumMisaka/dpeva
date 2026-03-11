import logging

from dpeva.utils.logs import StreamToLogger


def test_stream_to_logger_flushes_partial_line():
    records = []
    logger = logging.getLogger("test_stream_logger")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger.addHandler(_Handler())

    stream = StreamToLogger(logger, logging.INFO)
    stream.write("hello")
    assert records == []
    stream.flush()
    assert records == ["hello"]


def test_stream_to_logger_splits_lines():
    records = []
    logger = logging.getLogger("test_stream_logger_lines")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger.addHandler(_Handler())

    stream = StreamToLogger(logger, logging.INFO)
    stream.write("line1\nline2\n")
    stream.flush()
    assert records == ["line1", "line2"]
