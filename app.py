# streamlit_audio_recorder by stefanrmmr (rs. analytics) - version January 2023

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from predict import create_spectrogram, Predict, DATA_FOLDER
import time
import os

# DESIGN implement changes to the standard streamlit UI/UX
# --> optional, not relevant for the functionality of the component!
st.set_page_config(page_title="streamlit_audio_recorder")
# Design move app further up and remove top padding
st.markdown(
    """<style>.css-1egvi7u {margin-top: -3rem;}</style>""", unsafe_allow_html=True
)
# Design change st.Audio to fixed height of 45 pixels
st.markdown("""<style>.stAudio {height: 45px;}</style>""", unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown(
    """<style>.css-v37k9u a {color: #ff4c4b;}</style>""", unsafe_allow_html=True
)  # darkmode
st.markdown(
    """<style>.css-nlntq9 a {color: #ff4c4b;}</style>""", unsafe_allow_html=True
)  # lightmode


def audiorec_demo_app():

    # TITLE and Creator information
    st.title("🎤 Streamlit audio recorder")

    st.write("\n\n")

    # TUTORIAL: How to use STREAMLIT AUDIO RECORDER?
    # by calling this function an instance of the audio recorder is created
    # once a recording is completed, audio data will be saved to wav_audio_data
    wav_audio_data = None
    option = st.selectbox("Select an option", ("Record", "Upload"), index=None)
    if option == "Upload":
        upload_file = st.file_uploader(
            "Choose an audio file", type=["wav"], accept_multiple_files=False
        )
        if upload_file is not None:
            wav_audio_data = upload_file.read()
    else:
        wav_audio_data = audio_recorder(
            key=123, icon_size="2x"
        )  # tadaaaa! yes, that's it! :D

    # add some spacing and informative messages
    col_info, col_space = st.columns([1.5, 0.43])
    with col_info:
        st.write("\n")  # add vertical spacer
        st.write("\n")  # add vertical spacer
        st.write(
            "The .wav audio data, as received in the backend Python code,"
            " will be displayed below this message as soon as it has"
            " been processed. 🎈"
        )
    # if st.button("Reset", type="primary"):
    #     wav_audio_data = None
    #     st.rerun()

    if wav_audio_data is not None:
        # display audio data as received on the Python side
        col_playback, col_space = st.columns([0.58, 0.42])
        with col_playback:
            st.audio(wav_audio_data, format="audio/wav")
        if st.button("Show Result"):
            st.markdown("### List of functions:")
            predict(wav_audio_data)
        # st.markdown("1. Command detection")
        # if st.button("Predict", key=1):
        #     command_detection(wav_audio_data, file_name)
        # st.markdown("2. Speaker verification")
        # if st.button("Predict", key=2):
        #     speaker_verifcation(wav_audio_data, file_name)
        # st.markdown("3. Fake voice recognization")
        # if st.button("Predict", key=3):
        #     fake_voice(wav_audio_data, file_name)


def predict(wav_audio_data):
    current_time_seconds = time.time()
    file_name = str(int(current_time_seconds * 1000))
    create_spectrogram(wav_audio_data, file_name)

    task1 = Predict(task=1, name=file_name)
    st.markdown("1. Command detection: " + task1)

    task2 = Predict(task=2, name=file_name)
    st.markdown("2. Speaker verification: " + task2)

    task3 = Predict(task=3, name=file_name)
    st.markdown("3. Fake voice recognization: " + task3)


def speaker_verifcation(wav_audio_data, file_name):
    # create_spectrogram(wav_audio_data, file_name)
    result = Predict(task=2, name=file_name)
    st.write("Result: " + result)


def command_detection(wav_audio_data, file_name):
    # create_spectrogram(wav_audio_data, file_name)
    result = Predict(task=1, name=file_name)
    st.write("Result: " + result)


def fake_voice(wav_audio_data, file_name):
    # create_spectrogram(wav_audio_data, file_name)
    result = Predict(task=3, name=file_name)
    st.write("Result: " + result)


if __name__ == "__main__":
    # call main function
    audiorec_demo_app()
