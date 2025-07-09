#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動化工具模塊

此模塊提供自動化功能，例如鍵盤和滑鼠模擬。
"""

import time
import logging
from typing import Any, Dict, Optional, Union

# 設置日誌
logger = logging.getLogger("AutomationUtils")

# 嘗試導入pynput庫
try:
    from pynput.keyboard import Key, Controller as KeyboardController
    KEYBOARD_AVAILABLE = True
except ImportError:
    logger.warning("無法導入pynput庫，鍵盤模擬功能將不可用。請安裝: pip install pynput")
    KEYBOARD_AVAILABLE = False
    # 創建空的Key類用於類型註解
    Key = None

# 鍵盤控制器實例（懶加載）
_keyboard_controller = None

def get_keyboard_controller() -> Optional[Any]:
    """
    獲取鍵盤控制器實例（懶加載）
    
    返回：
        KeyboardController或None: 鍵盤控制器實例，如果pynput不可用則返回None
    """
    global _keyboard_controller
    
    if not KEYBOARD_AVAILABLE:
        return None
    
    if _keyboard_controller is None:
        try:
            _keyboard_controller = KeyboardController()
            logger.info("鍵盤控制器已初始化")
        except Exception as e:
            logger.error(f"初始化鍵盤控制器時出錯: {e}")
            return None
    
    return _keyboard_controller

def simulate_key_action(key_obj: Union[str, Any], action_type: str = "tap", delay: float = 0.05) -> bool:
    """
    模擬鍵盤按鍵動作
    
    參數:
        key_obj: 要模擬的按鍵，可以是pynput.keyboard.Key的成員或單字符字串
        action_type: 按鍵動作類型，可以是"tap"（按下並釋放）、"press"（僅按下）或"release"（僅釋放）
        delay: 按下和釋放之間的延遲（秒）
        
    返回:
        bool: 操作是否成功
    """
    if not KEYBOARD_AVAILABLE:
        logger.error("pynput庫不可用，無法模擬鍵盤動作")
        return False
    
    keyboard = get_keyboard_controller()
    if not keyboard:
        return False
    
    # 如果是字符串，轉換為適當的格式
    if isinstance(key_obj, str):
        if key_obj.lower() == "space":
            key_obj = Key.space
        elif key_obj.lower() == "enter":
            key_obj = Key.enter
        elif key_obj.lower() == "tab":
            key_obj = Key.tab
        elif len(key_obj) != 1:
            logger.error(f"無效的按鍵字符串: {key_obj}. 應為單個字符或特殊鍵名稱")
            return False
    
    try:
        if action_type == "tap":
            keyboard.press(key_obj)
            time.sleep(delay)  # 微小延遲，確保事件註冊
            keyboard.release(key_obj)
            logger.info(f"成功模擬按鍵點擊: {key_obj}")
        elif action_type == "press":
            keyboard.press(key_obj)
            logger.info(f"成功模擬按鍵按下: {key_obj}")
        elif action_type == "release":
            keyboard.release(key_obj)
            logger.info(f"成功模擬按鍵釋放: {key_obj}")
        else:
            logger.error(f"無效的按鍵動作類型: {action_type}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"模擬按鍵動作失敗: {key_obj} - {e}")
        return False

def execute_automation_action(action: Dict[str, Any]) -> bool:
    """
    執行自動化動作
    
    參數:
        action: 包含動作信息的字典，必須包含'type'字段
        
    返回:
        bool: 操作是否成功
    """
    action_type = action.get('type')
    
    if not action_type:
        logger.error("動作缺少必要的'type'字段")
        return False
    
    if action_type == "keyboard_action":
        key = action.get('key', 'space')  # 默認為空格鍵
        key_action = action.get('key_action', 'tap')  # 默認為點擊
        delay = action.get('delay', 0.05)  # 默認延遲
        return simulate_key_action(key, key_action, delay)
    else:
        logger.error(f"不支持的動作類型: {action_type}")
        return False 