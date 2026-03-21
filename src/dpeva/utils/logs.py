import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from dpeva.constants import DEFAULT_LOG_MAX_BYTES, DEFAULT_LOG_BACKUP_COUNT

def setup_workflow_logger(logger_name: str, work_dir: str, filename: str, capture_stdout: bool = True):
    """
    Configures a file handler for the specified logger.
    
    Args:
        logger_name (str): Name of the logger to configure.
        work_dir (str): Directory where the log file will be saved.
        filename (str): Name of the log file.
        capture_stdout (bool): Whether to capture stdout/stderr to this logger.
    """
    log_path = os.path.join(work_dir, filename)
    
    # Ensure work_dir exists
    os.makedirs(work_dir, exist_ok=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # File Handler
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=DEFAULT_LOG_MAX_BYTES, 
        backupCount=DEFAULT_LOG_BACKUP_COUNT
    )
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Avoid duplicate handlers
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == log_path for h in logger.handlers):
        logger.addHandler(file_handler)
    
    # Optional: Capture stdout/stderr (useful for Slurm jobs)
    if capture_stdout:
        logger.propagate = False
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        if not buf:
            return
        self.linebuf += buf
        while "\n" in self.linebuf:
            line, self.linebuf = self.linebuf.split("\n", 1)
            if line:
                self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        if self.linebuf:
            self.logger.log(self.log_level, self.linebuf.rstrip())
            self.linebuf = ''

def close_workflow_logger(logger_name: str, log_path: str):
    """
    关闭并移除指定路径的文件日志 Handler。
    
    Args:
        logger_name: 日志器名称
        log_path: 日志文件绝对路径
    """
    logger = logging.getLogger(logger_name)
    abs_path = os.path.abspath(log_path)
    
    handlers_to_remove = []
    for h in logger.handlers:
        if hasattr(h, 'baseFilename') and h.baseFilename == abs_path:
            handlers_to_remove.append(h)
            
    for h in handlers_to_remove:
        h.close()
        logger.removeHandler(h)
