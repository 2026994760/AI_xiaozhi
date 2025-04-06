import os
import queue
import wave
import pyaudio
import pygame
import time
import numpy as np
import threading
import uuid
import re
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit

from openai import OpenAI
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer
from dashscope.audio.asr import *
from flask_cors import CORS

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化pygame音频
pygame.mixer.init()

# 设置API Key
dashscope.api_key = ""
deepseek_api_key = ""
# client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
client = OpenAI(api_key=dashscope.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


# 语音合成配置
model = "cosyvoice-v1"
voice = "longmiao"

# 创建必要的目录
AUDIO_DIR = "audio_files"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# 创建队列和事件
text_queue = queue.Queue()
audio_queue = queue.Queue()
stop_event = threading.Event()
play_stop_event = threading.Event()

def is_valid_text(text):
    """检查文本是否有效"""
    text = text.strip()
    if not text:
        return False
    if re.match(r'^[\s\. , ! ? ，。！？]+$', text):
        return False
    return True

def process_text():
    """处理文本并生成语音的后台线程"""
    while not stop_event.is_set():
        try:
            text = text_queue.get(timeout=1)
            if text and is_valid_text(text):
                try:
                    synthesizer = SpeechSynthesizer(model=model, voice=voice, speech_rate=1.2)
                    audio = synthesizer.call(text)
                    if audio and isinstance(audio, bytes):
                        filename = f"{AUDIO_DIR}/语音_{uuid.uuid4()}.mp3"
                        with open(filename, 'wb') as f:
                            f.write(audio)
                        audio_queue.put(filename)
                    else:
                        print(f"语音合成失败: {text}")
                except Exception as e:
                    print(f"语音合成出错: {e}")
            else:
                print(f"跳过无效文本: {text}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"处理文本时出错: {e}")

def play_audio():
    """播放音频的后台线程"""
    while not play_stop_event.is_set():
        try:
            audio_file = audio_queue.get(timeout=3)
            if audio_file and os.path.exists(audio_file):
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                except Exception as e:
                    print(f"播放音频时出错: {e}")
                finally:
                    try:
                        os.remove(audio_file)
                    except:
                        pass
        except queue.Empty:
            continue
        except Exception as e:
            print(f"播放音频时出错: {e}")

def record_audio(output_file="recorded_audio.wav", silence_threshold=500, silence_duration=1.0):
    """录制音频，通过检测语音停顿自动结束"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("开始录音，请说话...")
    print("检测到长时间静音将自动结束录音")

    frames = []
    silence_frames = 0
    silence_threshold_frames = int(silence_duration * RATE / CHUNK)
    is_speaking = False

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        
        audio_data = np.frombuffer(data, dtype=np.int16)
        energy = np.sum(np.abs(audio_data)) / len(audio_data)
        
        if energy > silence_threshold:
            is_speaking = True
            silence_frames = 0
        elif is_speaking:
            silence_frames += 1
            if silence_frames >= silence_threshold_frames:
                print("检测到静音，录音结束")
                break

    print("录音结束")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return output_file

def speech_recognition(audio_file_path):
    """对音频文件进行语音识别"""
    class Callback(TranslationRecognizerCallback):
        def __init__(self):
            self.final_result = ""
            self.is_complete = False

        def on_event(self, request_id, transcription_result, translation_result, usage):
            if transcription_result is not None:
                self.final_result = transcription_result.text
                print(f"识别结果: {self.final_result}")

        def on_complete(self):
            self.is_complete = True

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_file_path}")

    callback = Callback()
    translator = TranslationRecognizerChat(
        model="gummy-chat-v1",
        format="wav",
        sample_rate=16000,
        callback=callback,
    )

    translator.start()

    try:
        with open(audio_file_path, 'rb') as f:
            if os.path.getsize(audio_file_path):
                while True:
                    audio_data = f.read(12800)
                    if not audio_data:
                        break
                    if not translator.send_audio_frame(audio_data):
                        break
            else:
                raise Exception('音频文件为空')
    finally:
        translator.stop()

    while not callback.is_complete:
        time.sleep(0.1)

    return callback.final_result

# 启动处理线程
process_thread = threading.Thread(target=process_text)
process_thread.start()

# 启动播放线程
play_thread = threading.Thread(target=play_audio)
play_thread.start()

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    """语音对话API接口"""
    try:
        # 1. 录音
        audio_file = record_audio()
        
        # 2. 语音识别
        question = speech_recognition(audio_file)
        if not question:
            return jsonify({"error": "未能识别到语音"}), 400
        
        # 3. 调用LLM生成回复
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你扮演的角色:女友小智;你的主人and我的角色:余训尧;你的人物性格:温柔,体贴,善解人意但却不失可爱;"},
                {"role": "user", "content": "主人余训尧的问题如下:" + question},
            ],
            stream=True
        )
        
        # 4. 处理回复并生成语音
        current_text = ''
        for chunk in response:
            current_text += chunk.choices[0].delta.content
            if chunk.choices[0].delta.content and chunk.choices[0].delta.content[-1] in ['！', '？', '。', '，', '!', '?', ','] and current_text != '':
                text_queue.put(current_text)
                print(f"发送消息: {current_text}")
                socketio.emit('message', {'type': 'text', 'content': current_text})
                current_text = ''
        
        if current_text and len(current_text) > 1:
            text_queue.put(current_text)
            socketio.emit('message', {'type': 'text', 'content': current_text})
        
        socketio.emit('message', {'type': 'end'})
        # print(jsonify({
        #     "status": "success",
        #     "question": question,
        #     "message": "正在生成回复"
        #     }))
        return jsonify({
            "status": "success",
            "question": question,
            "message": "正在生成回复"
        })
        
    except Exception as e:
        print(f"Error in voice_chat: {str(e)}")  # 添加错误日志
        return jsonify({"error": str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop():
    """停止API接口"""
    try:
        stop_event.set()
        play_stop_event.set()
        process_thread.join()
        play_thread.join()
        return jsonify({"status": "success", "message": "已停止所有处理"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)