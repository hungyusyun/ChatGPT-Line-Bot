from typing import List, Dict
import requests


class ModelInterface:
    def check_token_valid(self) -> bool:
        pass

    def chat_completions(self, messages: List[Dict], model_engine: str) -> str:
        pass

    def audio_transcriptions(self, file, model_engine: str) -> str:
        pass

    def image_generations(self, prompt: str, model_engine: str) -> str:
        pass


class OpenAIModel(ModelInterface):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.openai.com/v1'

    def _request(self, method, endpoint, body=None, files=None):
        self.headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        try:
            if method == 'GET':
                r = requests.get(f'{self.base_url}{endpoint}', headers=self.headers)
            elif method == 'POST':
                if body:
                    self.headers['Content-Type'] = 'application/json'
                r = requests.post(f'{self.base_url}{endpoint}', headers=self.headers, json=body, files=files)
            r = r.json()
            if r.get('error'):
                return False, None, r.get('error', {}).get('message')
        except Exception:
            return False, None, 'OpenAI API 系統不穩定，請稍後再試'
        return True, r, None

    def check_token_valid(self):
        return self._request('GET', '/models')

    def chat_completions(self, messages, model_engine) -> str:
        json_body = {
            'model': model_engine,
            'messages': messages
        }
        return self._request('POST', '/chat/completions', body=json_body)

    def audio_transcriptions(self, file_path, model_engine) -> str:
        with open(file_path, 'rb') as f:
            files = {
                'file': f,
                'model': (None, model_engine),
            }
            return self._request('POST', '/audio/transcriptions', files=files)

    def image_generations(self, prompt: str, model_engine: str) -> str:
        json_body = {
            'model': model_engine,
            'prompt': prompt,
            'n': 1,
            'size': '1024x1024'
        }
        if model_engine.startswith('gpt-image'):
            # gpt-image 系列只回傳 base64；用低畫質 JPEG 讓檔案小於 LINE 預覽圖 1MB 上限
            json_body['quality'] = 'low'
            json_body['output_format'] = 'jpeg'
        return self._request('POST', '/images/generations', body=json_body)
