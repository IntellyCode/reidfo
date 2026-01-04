import pytest
import logging

from reidfo.logging import setup_logger, set_logging_level


class TestSetupLogger:
    def test_returns_logger(self):
        logger = setup_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self):
        logger = setup_logger("my_test_logger")
        assert logger.name == "my_test_logger"

    def test_logger_has_handler(self):
        logger = setup_logger("test_with_handler")
        assert len(logger.handlers) >= 1

    def test_handler_is_stream_handler(self):
        logger = setup_logger("test_stream_handler")
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler) for h in logger.handlers
        )
        assert has_stream_handler

    def test_does_not_duplicate_handlers(self):
        logger_name = "test_no_duplicate"
        logger1 = setup_logger(logger_name)
        initial_count = len(logger1.handlers)

        logger2 = setup_logger(logger_name)
        assert len(logger2.handlers) == initial_count

    def test_formatter_format(self):
        logger = setup_logger("test_formatter")
        handler = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler))
        formatter = handler.formatter
        assert formatter is not None
        assert "%(asctime)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt


class TestSetLoggingLevel:
    def test_sets_level_for_reidfo_loggers(self):
        logger = setup_logger("reidfo.test_module")
        set_logging_level(logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_sets_level_info(self):
        logger = setup_logger("reidfo.test_info")
        set_logging_level(logging.INFO)
        assert logger.level == logging.INFO

    def test_sets_level_warning(self):
        logger = setup_logger("reidfo.test_warning")
        set_logging_level(logging.WARNING)
        assert logger.level == logging.WARNING

    def test_sets_level_error(self):
        logger = setup_logger("reidfo.test_error")
        set_logging_level(logging.ERROR)
        assert logger.level == logging.ERROR

    def test_does_not_affect_non_reidfo_loggers(self):
        other_logger = logging.getLogger("other_package.module")
        other_logger.setLevel(logging.WARNING)

        set_logging_level(logging.DEBUG)
        assert other_logger.level == logging.WARNING