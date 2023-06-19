
import streamlit as st

import cv2

import numpy as np



def main():

    st.title("Video Streaming App")

    st.sidebar.title("Settings")

    st.set_option('deprecation.showfileUploaderEncoding', False)



    # Read video from local file

    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi"])



    if uploaded_file is not None:

        # Get video properties

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        cap = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



        # Define codec and create VideoWriter object

        fourcc = cv2.VideoWriter_fourcc(*'H264')

        out = cv2.VideoWriter('rtsp://localhost:8554/stream', fourcc, fps, (width, height))



        # Loop through frames and write to RTSP stream

        while True:

            ret, frame = cap.read()

            if ret:

                out.write(frame)

                st.image(frame, channels="BGR")

            else:

                break



        # Release resources

        cap.release()

        out.release()



if __name__ == '__main__':

    main()
