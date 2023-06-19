import cv2
import streamlit as st

def main():
    st.title("RTSP Video Stream")
    st.write("Enter the RTSP URL below:")
    rtsp_url = st.text_input("RTSP URL")
    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            st.error("Error opening video stream or file")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, channels="RGB")
        cap.release()

if __name__ == "__main__":
    main()