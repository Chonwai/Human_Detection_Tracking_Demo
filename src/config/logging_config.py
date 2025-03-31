#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日誌配置文件
設置系統的日誌記錄格式和級別
"""
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: Path = None) -> logging.Logger:
    """
    配置日誌系統
    
    Args:
        log_dir: 日誌文件存儲目錄
        
    Returns:
        logging.Logger: 配置好的日誌器實例
    """
    # 創建日誌目錄
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建日誌文件名
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日誌格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置文件處理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 配置控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 創建日誌器
    logger = logging.getLogger("HumanDetectionTracking")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 