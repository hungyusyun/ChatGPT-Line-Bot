from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, AudioMessage
)
import base64
import glob
import os
import uuid

from src.models import OpenAIModel
from src.memory import Memory
from src.logger import logger
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader

load_dotenv('.env')

CHAT_MODEL = os.getenv('OPENAI_MODEL_ENGINE') or 'gpt-5-mini'
IMAGE_MODEL = os.getenv('OPENAI_IMAGE_MODEL') or 'gpt-image-1'
TRANSCRIBE_MODEL = os.getenv('OPENAI_TRANSCRIBE_MODEL') or 'gpt-4o-mini-transcribe'
ALLOWED_USER_IDS = {uid.strip() for uid in (os.getenv('ALLOWED_USER_IDS') or '').split(',') if uid.strip()}
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images')
MAX_KEEP_IMAGES = 20

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
youtube = Youtube(step=4)
website = Website()

memory = Memory(system_message=os.getenv('SYSTEM_MESSAGE'), memory_message_count=2)
model = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY'))


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'


def check_permission(event):
    """
    回傳 True 代表可以繼續處理。
    尚未設定 ALLOWED_USER_IDS 時進入設定模式：回覆對方的 user_id，但不呼叫 OpenAI。
    已設定但不在名單內的人，直接不理。
    """
    user_id = event.source.user_id
    if not ALLOWED_USER_IDS:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text=f'尚未設定白名單。\n你的 user_id 是：\n{user_id}\n\n請把它填進環境變數 ALLOWED_USER_IDS 後重新啟動。'))
        return False
    if user_id not in ALLOWED_USER_IDS:
        logger.info(f'Ignored message from non-allowed user: {user_id}')
        return False
    return True


def save_image(b64_data):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    file_name = f'{uuid.uuid4()}.jpg'
    with open(os.path.join(IMAGE_DIR, file_name), 'wb') as f:
        f.write(base64.b64decode(b64_data))
    # 只保留最近幾張，避免硬碟被塞滿
    old_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')), key=os.path.getmtime)[:-MAX_KEEP_IMAGES]
    for path in old_files:
        os.remove(path)
    base_url = os.getenv('PUBLIC_BASE_URL') or request.url_root.replace('http://', 'https://')
    return f"{base_url.rstrip('/')}/static/images/{file_name}"


def get_error_message(e):
    if str(e).startswith('Incorrect API key provided'):
        return 'OpenAI API Key 有誤，請檢查環境變數 OPENAI_API_KEY。'
    if str(e).startswith('That model is currently overloaded with other requests.'):
        return '已超過負荷，請稍後再試'
    return str(e)


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    if not check_permission(event):
        return
    user_id = event.source.user_id
    text = event.message.text.strip()
    logger.info(f'{user_id}: {text}')

    try:
        if text.startswith('/指令說明'):
            msg = TextSendMessage(text="指令：\n/系統訊息 + Prompt\n👉 Prompt 可以命令機器人扮演某個角色，例如：請你扮演擅長做總結的人\n\n/清除\n👉 當前每一次都會紀錄最後兩筆歷史紀錄，這個指令能夠清除歷史訊息\n\n/圖像 + Prompt\n👉 以文字生成圖像\n\n語音輸入\n👉 先將語音轉換成文字，再以文字回覆\n\n貼上 YouTube 或新聞網址\n👉 自動總結內容\n\n其他文字輸入\n👉 調用 ChatGPT 以文字回覆")

        elif text.startswith('/系統訊息'):
            memory.change_system_message(user_id, text[5:].strip())
            msg = TextSendMessage(text='輸入成功')

        elif text.startswith('/清除'):
            memory.remove(user_id)
            msg = TextSendMessage(text='歷史訊息清除成功')

        elif text.startswith('/圖像'):
            prompt = text[3:].strip()
            memory.append(user_id, 'user', prompt)
            is_successful, response, error_message = model.image_generations(prompt, IMAGE_MODEL)
            if not is_successful:
                raise Exception(error_message)
            image = response['data'][0]
            url = image.get('url') or save_image(image['b64_json'])
            msg = ImageSendMessage(
                original_content_url=url,
                preview_image_url=url
            )
            memory.append(user_id, 'assistant', url)

        else:
            memory.append(user_id, 'user', text)
            url = website.get_url_from_text(text)
            if url:
                if youtube.retrieve_video_id(text):
                    is_successful, chunks, error_message = youtube.get_transcript_chunks(youtube.retrieve_video_id(text))
                    if not is_successful:
                        raise Exception(error_message)
                    youtube_transcript_reader = YoutubeTranscriptReader(model, CHAT_MODEL)
                    is_successful, response, error_message = youtube_transcript_reader.summarize(chunks)
                    if not is_successful:
                        raise Exception(error_message)
                    role, response = get_role_and_content(response)
                    msg = TextSendMessage(text=response)
                else:
                    chunks = website.get_content_from_url(url)
                    if len(chunks) == 0:
                        raise Exception('無法撈取此網站文字')
                    website_reader = WebsiteReader(model, CHAT_MODEL)
                    is_successful, response, error_message = website_reader.summarize(chunks)
                    if not is_successful:
                        raise Exception(error_message)
                    role, response = get_role_and_content(response)
                    msg = TextSendMessage(text=response)
            else:
                is_successful, response, error_message = model.chat_completions(memory.get(user_id), CHAT_MODEL)
                if not is_successful:
                    raise Exception(error_message)
                role, response = get_role_and_content(response)
                msg = TextSendMessage(text=response)
            memory.append(user_id, role, response)
    except Exception as e:
        memory.remove(user_id)
        msg = TextSendMessage(text=get_error_message(e))
    line_bot_api.reply_message(event.reply_token, msg)


@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    if not check_permission(event):
        return
    user_id = event.source.user_id
    audio_content = line_bot_api.get_message_content(event.message.id)
    input_audio_path = f'{str(uuid.uuid4())}.m4a'
    with open(input_audio_path, 'wb') as fd:
        for chunk in audio_content.iter_content():
            fd.write(chunk)

    try:
        is_successful, response, error_message = model.audio_transcriptions(input_audio_path, TRANSCRIBE_MODEL)
        if not is_successful:
            raise Exception(error_message)
        memory.append(user_id, 'user', response['text'])
        is_successful, response, error_message = model.chat_completions(memory.get(user_id), CHAT_MODEL)
        if not is_successful:
            raise Exception(error_message)
        role, response = get_role_and_content(response)
        memory.append(user_id, role, response)
        msg = TextSendMessage(text=response)
    except Exception as e:
        memory.remove(user_id)
        msg = TextSendMessage(text=get_error_message(e))
    finally:
        os.remove(input_audio_path)
    line_bot_api.reply_message(event.reply_token, msg)


@app.route("/", methods=['GET'])
def home():
    return 'Hello World'


if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning('OPENAI_API_KEY is not set')
    app.run(host='0.0.0.0', port=8080)
