import json
import logging
from typing import Any, Dict, List, Tuple, Optional, IO

# 從專案的日誌設定獲取 logger
# 假設 src/config/logging_config.py 中有一個 get_logger 函數
# 如果沒有，可以先用 logging.getLogger(__name__) 替代
# from src.config.logging_config import get_logger
# logger = get_logger(__name__)
# 臨時替代：
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # 臨時基本配置

# 預期的操作符列表
VALID_OPERATORS = ['==', '>=', '<=', '>', '<']
VALID_TRIGGER_TYPES = ['person_count']
# 可選：預期的 Cue 分類 (如果設計中包含)
# VALID_CUE_CATEGORIES = ['Lighting', 'Sound', 'Video', 'General']

def _validate_script_format(script_data: List[Dict[str, Any]], script_source_name: str = "script") -> Tuple[bool, str]:
    """
    (內部輔助函數) 驗證解析後的劇本數據列表是否符合預期結構和邏輯。

    Args:
        script_data: 解析後的劇本事件列表 (每個事件是一個字典)。
        script_source_name: 用於錯誤訊息中標識劇本來源 (例如檔案名)。

    Returns:
        Tuple[bool, str]: (is_valid, detailed_error_message_string)
                          is_valid 為 True 表示驗證通過，錯誤訊息為空字串。
                          is_valid 為 False 表示驗證失敗，錯誤訊息包含所有檢測到的問題。
    """
    errors: List[str] = []
    if not isinstance(script_data, list):
        return False, f"劇本頂層結構必須是一個列表 (list)，但得到的是 {type(script_data).__name__}。"

    for i, event in enumerate(script_data):
        event_identifier = f"事件 {i+1} (ID: {event.get('event_id', '未提供')}, 描述: \\'{event.get('description', '未提供')[:20]}...\\')"
        
        if not isinstance(event, dict):
            errors.append(f"{event_identifier}: 每個事件必須是一個字典 (dict)，但得到的是 {type(event).__name__}。")
            continue # 後續基於字典的檢查無意義

        # 1. 檢查必要欄位
        required_fields = ['time_start', 'time_end', 'trigger_condition', 'predicted_cues']
        for field in required_fields:
            if field not in event:
                errors.append(f"{event_identifier}: 缺少必要欄位 '{field}'。")

        # 如果缺少關鍵結構欄位，後續檢查可能引發錯誤，可以選擇提前返回或跳過該事件的進一步檢查
        if any(field not in event for field in ['trigger_condition', 'predicted_cues']):
            # 為了收集更多錯誤，我們這裡選擇繼續，但要注意 None 檢查
            pass

        # 2. 檢查欄位類型和邏輯約束
        # time_start 和 time_end
        time_start = event.get('time_start')
        time_end = event.get('time_end')
        if not isinstance(time_start, (int, float)):
            errors.append(f"{event_identifier}: 'time_start' 應為數字 (int 或 float)，但得到的是 '{time_start}' (類型: {type(time_start).__name__})。")
        if not isinstance(time_end, (int, float)):
            errors.append(f"{event_identifier}: 'time_end' 應為數字 (int 或 float)，但得到的是 '{time_end}' (類型: {type(time_end).__name__})。")
        if isinstance(time_start, (int, float)) and isinstance(time_end, (int, float)) and time_start > time_end:
            errors.append(f"{event_identifier}: 'time_start' ({time_start}) 不能晚於 'time_end' ({time_end})。")

        # trigger_condition
        trigger_condition = event.get('trigger_condition')
        if not isinstance(trigger_condition, dict):
            errors.append(f"{event_identifier}: 'trigger_condition' 應為一個字典 (dict)，但得到的是 '{trigger_condition}' (類型: {type(trigger_condition).__name__})。")
        else:
            trigger_req_fields = ['type', 'operator', 'value']
            for field in trigger_req_fields:
                if field not in trigger_condition:
                    errors.append(f"{event_identifier}: 'trigger_condition' 中缺少必要欄位 '{field}'。")
            
            trigger_type = trigger_condition.get('type')
            if trigger_type not in VALID_TRIGGER_TYPES:
                errors.append(f"{event_identifier}: 'trigger_condition.type' 的值 '{trigger_type}' 無效。有效值為: {VALID_TRIGGER_TYPES}。")
            
            operator = trigger_condition.get('operator')
            if operator not in VALID_OPERATORS:
                errors.append(f"{event_identifier}: 'trigger_condition.operator' 的值 '{operator}' 無效。有效值為: {VALID_OPERATORS}。")

            value = trigger_condition.get('value')
            if not isinstance(value, int) or value < 0: # 通常人數不能為負
                errors.append(f"{event_identifier}: 'trigger_condition.value' 應為非負整數，但得到的是 '{value}' (類型: {type(value).__name__})。")

        # predicted_cues
        predicted_cues = event.get('predicted_cues')
        if not isinstance(predicted_cues, list):
            errors.append(f"{event_identifier}: 'predicted_cues' 應為一個列表 (list)，但得到的是 '{predicted_cues}' (類型: {type(predicted_cues).__name__})。")
        else:
            if not predicted_cues: # 允許 predicted_cues 為空列表嗎？根據需求決定，這裡假設允許
                # errors.append(f"{event_identifier}: 'predicted_cues' 列表不能為空。")
                pass
            for cue_idx, cue in enumerate(predicted_cues):
                cue_identifier = f"{event_identifier} 中的 Predicted Cue {cue_idx+1}"
                if not isinstance(cue, dict):
                    errors.append(f"{cue_identifier}: 每個 cue 必須是一個字典 (dict)，但得到的是 {type(cue).__name__}。")
                    continue
                
                cue_req_fields = ['offset', 'cue_description']
                for field in cue_req_fields:
                    if field not in cue:
                        errors.append(f"{cue_identifier}: 缺少必要欄位 '{field}'。")
                
                offset = cue.get('offset')
                if not isinstance(offset, (int, float)) or offset < 0:
                    errors.append(f"{cue_identifier}: 'offset' 應為非負數字，但得到的是 '{offset}' (類型: {type(offset).__name__})。")
                
                description = cue.get('cue_description')
                if not isinstance(description, str) or not description.strip():
                    errors.append(f"{cue_identifier}: 'cue_description' 應為非空字串，但得到的是 '{description}' (類型: {type(description).__name__})。")
                
                # 可選: 檢查 cue_category (如果最終設計包含此欄位)
                # cue_category = cue.get('cue_category')
                # if cue_category is not None and cue_category not in VALID_CUE_CATEGORIES:
                #     errors.append(f"{cue_identifier}: 'cue_category' 的值 '{cue_category}' 無效。有效值為: {VALID_CUE_CATEGORIES}。")

    if errors:
        # 將所有錯誤合併為一個字串，每條錯誤換行
        # 在錯誤訊息前加上來源
        error_header = f"劇本 '{script_source_name}' 驗證失敗，發現以下問題：\\n"
        return False, error_header + "\\n".join(f"- {err}" for err in errors)
    
    return True, ""


def load_script_from_uploaded_file(uploaded_file_obj: IO[Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    從 Streamlit UploadedFile 對象 (或任何 file-like object) 載入、解析並驗證劇本。

    Args:
        uploaded_file_obj: 一個 file-like object，例如 Streamlit 的 UploadedFile。
                           它應該提供 .name (用於錯誤報告) 和 .read() 方法。

    Returns:
        Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
            - (parsed_data, None) 如果成功。parsed_data 是事件字典的列表。
            - (None, error_message) 如果失敗。error_message 包含錯誤描述。
    """
    script_name = getattr(uploaded_file_obj, 'name', '未知檔案')
    logger.info(f"開始從檔案 '{script_name}' 載入劇本...")

    try:
        # 假設 uploaded_file_obj.read() 返回 bytes，需要 decode
        # 如果它直接返回 str，則不需要 decode
        # Streamlit UploadedFile.read() 返回 bytes
        script_content_bytes = uploaded_file_obj.read()
        script_content_str = script_content_bytes.decode('utf-8')
        logger.debug(f"從 '{script_name}' 讀取的原始內容 (前200字符): {script_content_str[:200]}")
    except Exception as e:
        logger.error(f"讀取檔案 '{script_name}' 失敗: {e}", exc_info=True)
        return None, f"讀取檔案 '{script_name}' 失敗: {e}"

    try:
        parsed_data: List[Dict[str, Any]] = json.loads(script_content_str)
        logger.info(f"檔案 '{script_name}' JSON 解析成功。")
    except json.JSONDecodeError as e:
        logger.error(f"檔案 '{script_name}' JSON 解析失敗: {e}", exc_info=True)
        # 提供更友好的錯誤提示，例如錯誤發生的行號和列號
        error_msg = f"劇本 '{script_name}' JSON 格式錯誤: {e.msg} (在第 {e.lineno} 行，第 {e.colno} 列附近)。"
        return None, error_msg

    is_valid, validation_errors = _validate_script_format(parsed_data, script_name)
    if not is_valid:
        logger.error(f"劇本 '{script_name}' 驗證失敗:\n{validation_errors}")
        return None, validation_errors
    
    logger.info(f"劇本 '{script_name}' 成功載入並通過驗證。共 {len(parsed_data)} 個事件。")
    return parsed_data, None

# 為了方便本地測試，可以添加一個從檔案路徑載入的函數 (可選)
def load_script_from_path(file_path: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    (可選的輔助函數) 從本地檔案路徑載入、解析並驗證劇本。
    主要用於開發和測試。
    """
    logger.info(f"開始從本地路徑 '{file_path}' 載入劇本...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 直接將 file-like object 傳給 load_script_from_uploaded_file 進行統一處理
            # 但這樣 file_path 不會作為 script_name 傳遞，除非修改 load_script_from_uploaded_file 的參數
            # 或者複製其邏輯，這裡選擇簡化，直接調用核心邏輯
            script_content_str = f.read()
            
        # 以下邏輯與 load_script_from_uploaded_file 中解析和驗證部分類似
        parsed_data: List[Dict[str, Any]] = json.loads(script_content_str)
        logger.info(f"檔案 '{file_path}' JSON 解析成功。")
        
        is_valid, validation_errors = _validate_script_format(parsed_data, file_path)
        if not is_valid:
            logger.error(f"劇本 '{file_path}' 驗證失敗:\n{validation_errors}")
            return None, validation_errors
        
        logger.info(f"劇本 '{file_path}' 成功載入並通過驗證。共 {len(parsed_data)} 個事件。")
        return parsed_data, None

    except FileNotFoundError:
        logger.error(f"劇本檔案 '{file_path}' 未找到。")
        return None, f"劇本檔案 '{file_path}' 未找到。"
    except json.JSONDecodeError as e:
        logger.error(f"檔案 '{file_path}' JSON 解析失敗: {e.msg} (行 {e.lineno} 列 {e.colno})", exc_info=True)
        return None, f"劇本 '{file_path}' JSON 格式錯誤: {e.msg} (在第 {e.lineno} 行，第 {e.colno} 列附近)。"
    except Exception as e:
        logger.error(f"從 '{file_path}' 載入劇本時發生未知錯誤: {e}", exc_info=True)
        return None, f"從 '{file_path}' 載入劇本時發生未知錯誤: {e}"

if __name__ == '__main__':
    # 簡易的本地測試代碼
    # 創建一些臨時的測試JSON檔案來進行測試
    
    # 測試案例1: 有效劇本
    valid_script_content = """
    [
      {
        "event_id": "E001",
        "description": "Opening Scene",
        "time_start": 0.0,
        "time_end": 15.5,
        "trigger_condition": { "type": "person_count", "operator": "==", "value": 1 },
        "predicted_cues": [
          { "offset": 5.0, "cue_description": "LX Cue 5" }
        ]
      }
    ]
    """
    with open("temp_valid_script.json", "w", encoding="utf-8") as f:
        f.write(valid_script_content)
    
    print("\\n--- 測試有效劇本 (temp_valid_script.json) ---")
    data, error = load_script_from_path("temp_valid_script.json")
    if error:
        print(f"載入失敗: {error}")
    else:
        print(f"載入成功! 事件數量: {len(data) if data else 0}")
        # print(data)

    # 測試案例2: 格式錯誤的JSON (例如，尾部多了逗號)
    invalid_json_format_content = """
    [
      {
        "event_id": "E001",
        "time_start": 0.0, // Error: trailing comma
      }
    ]
    """ # 上面json格式錯誤，修正一下
    invalid_json_format_content = """
    [
      {
        "event_id": "E001",
        "time_start": 0.0, 
        "time_end": 10.0
      }, 
    ]
    """ # 尾部逗號
    with open("temp_invalid_json.json", "w", encoding="utf-8") as f:
        f.write(invalid_json_format_content)

    print("\\n--- 測試格式錯誤的JSON (temp_invalid_json.json) ---")
    data, error = load_script_from_path("temp_invalid_json.json")
    if error:
        print(f"載入失敗 (符合預期):\\n{error}")
    else:
        print("載入成功 (非預期)。")


    # 測試案例3: 邏輯錯誤的劇本 (缺少必要欄位, time_start > time_end)
    invalid_logic_content = """
    [
      { 
        "description": "Missing time_start",
        "time_end": 10.0,
        "trigger_condition": { "type": "person_count", "operator": "==", "value": 1 },
        "predicted_cues": [{ "offset": 1.0, "cue_description": "Cue A"}]
      },
      {
        "event_id": "E002",
        "time_start": 20.0,
        "time_end": 10.0, 
        "trigger_condition": { "type": "person_count", "operator": "==", "value": 1 },
        "predicted_cues": [{ "offset": 1.0, "cue_description": "Cue B"}]
      },
      {
        "event_id": "E003",
        "time_start": 30.0,
        "time_end": 40.0,
        "trigger_condition": { "type": "invalid_type", "operator": "!=", "value": "abc" },
        "predicted_cues": [{ "offset": -5.0, "cue_description": "" }]
      }
    ]
    """
    with open("temp_invalid_logic.json", "w", encoding="utf-8") as f:
        f.write(invalid_logic_content)

    print("\\n--- 測試邏輯錯誤的劇本 (temp_invalid_logic.json) ---")
    data, error = load_script_from_path("temp_invalid_logic.json")
    if error:
        print(f"載入失敗 (符合預期):\\n{error}")
    else:
        print("載入成功 (非預期)。")

    # 測試案例4: 頂層不是列表
    invalid_top_level_content = """
    { "message": "This is not a list" }
    """
    with open("temp_invalid_top_level.json", "w", encoding="utf-8") as f:
        f.write(invalid_top_level_content)
    print("\\n--- 測試頂層結構錯誤的劇本 (temp_invalid_top_level.json) ---")
    data, error = load_script_from_path("temp_invalid_top_level.json")
    if error:
        print(f"載入失敗 (符合預期):\\n{error}")
    else:
        print("載入成功 (非預期)。")

    print("\\n--- 清理臨時測試檔案 ---")
    import os
    try:
        os.remove("temp_valid_script.json")
        os.remove("temp_invalid_json.json")
        os.remove("temp_invalid_logic.json")
        os.remove("temp_invalid_top_level.json")
        print("臨時檔案已刪除。")
    except OSError as e:
        print(f"刪除臨時檔案錯誤: {e}")
    
    print("\\n--- 本地測試結束 ---") 