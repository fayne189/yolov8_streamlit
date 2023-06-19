import queue
import subprocess
import threading
import time
import streamlit as st
import numpy as np
import cv2
import asyncio

# 创建一个队列，用于存储画面数据
q = queue.Queue()

# 定义一个函数，用于从子进程中获取画面数据并存入队列中
def get_frame(q):
    cap = cv2.VideoCapture('video.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        q.put(frame)
    cap.release()

# 启动子进程，获取画面数据
t = threading.Thread(target=get_frame, args=(q,))
t.start()

# 在 Streamlit 中显示画面数据
stframe = st.empty()
async def update_frame():
    while True:
        if not q.empty():
            frame = q.get()
            stframe.image(frame, channels="BGR")
        await asyncio.sleep(0.01)
asyncio.create_task(update_frame())
