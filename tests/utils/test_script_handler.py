import unittest
import json
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

# 假設 script_handler.py 在上一級目錄的 src/utils/ 下
# import sys
# sys.path.append('../..') # 調整路徑以便導入
from src.utils.script_handler import load_script_from_uploaded_file, _validate_script_format # (如果_validate_script_format也想單獨測)

class TestScriptHandler(unittest.TestCase):

    def _create_mock_uploaded_file(self, content: str, name: str = "test_script.json") -> BytesIO:
        return BytesIO(content.encode('utf-8'))

    def test_load_valid_script(self):
        valid_content = """
        [
            {
                "event_id": "E001", "description": "Valid Event", "time_start": 0, "time_end": 10,
                "trigger_condition": {"type": "person_count", "operator": "==", "value": 1},
                "predicted_cues": [{"offset": 1, "cue_description": "Test Cue"}]
            }
        ]
        """
        mock_file = self._create_mock_uploaded_file(valid_content)
        setattr(mock_file, 'name', 'valid_script.json') # 模擬 UploadedFile 的 name 屬性
        
        data, error = load_script_from_uploaded_file(mock_file)
        
        self.assertIsNone(error, f"載入有效劇本時不應有錯誤，但得到: {error}")
        self.assertIsNotNone(data)
        if data: # 為了類型檢查
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]['event_id'], "E001")

    def test_load_script_missing_time_start(self):
        invalid_content = """
        [
            {
                "event_id": "E002", "time_end": 10, 
                "trigger_condition": {"type": "person_count", "operator": "==", "value": 1},
                "predicted_cues": [{"offset": 1, "cue_description": "Test Cue"}]
            }
        ]
        """
        mock_file = self._create_mock_uploaded_file(invalid_content, name="missing_field.json")
        setattr(mock_file, 'name', 'missing_field.json')

        data, error = load_script_from_uploaded_file(mock_file)
        self.assertIsNone(data, "資料應為 None 因為劇本無效")
        self.assertIsNotNone(error)
        if error: # 為了類型檢查
            self.assertIn("缺少必要欄位 'time_start'", error)
            self.assertIn("事件 1 (ID: E002", error) # 檢查錯誤訊息是否定位到事件

    # ... 添加更多測試案例 ...
    # - JSON 格式錯誤
    # - time_start > time_end
    # - trigger_condition.operator 無效
    # - trigger_condition.value 非數字
    # - predicted_cues.offset 為負數
    # - predicted_cues.cue_description 為空
    # - 頂層結構不是列表

if __name__ == '__main__':
    unittest.main()
