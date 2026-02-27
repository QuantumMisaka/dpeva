import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_workflow_logger(
    logger_name: str,
    work_dir: str,
    filename: str,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    capture_stdout: bool = False
) -> logging.Logger:
    """
    配置标准化的工作流文件日志。
    
    Args:
        logger_name: 日志器名称 (通常为 "dpeva" 或 __name__)
        work_dir: 日志文件存储目录
        filename: 日志文件名
        level: 日志级别
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的历史日志文件数量
        capture_stdout: 若为 True，则设 propagate=False，阻止日志向上传播到 stdout
    """
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, filename)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # 传播控制
    if capture_stdout:
        logger.propagate = False
    
    # 避免重复添加 Handler
    for h in logger.handlers:
        # Check baseFilename for FileHandler/RotatingFileHandler
        if hasattr(h, 'baseFilename') and h.baseFilename == os.path.abspath(log_path):
            return logger

    # 配置轮转 Handler
    handler = RotatingFileHandler(
        log_path, 
        mode='a', 
        maxBytes=max_bytes, 
        backupCount=backup_count, 
        encoding='utf-8'
    )
    
    # 标准格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)
    
    logger.addHandler(handler)
    logger.info(f"Log initialized: {log_path} (Rotation: {max_bytes/1024/1024:.1f}MB x {backup_count})")
    
    return logger

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
