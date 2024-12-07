import streamlit as st
import tensorflow as tf
import librosa
import soundfile as sf
from data_tools import scaled_in, inv_scaled_ou
from data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio

def prediction(weights_path, name_model, audio_file, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft):
    """ This function takes an input file, applies the model for denoising, and returns the noisy and denoised audio. """

    # Load the entire model (architecture + weights) from the .keras file
    loaded_model = tf.keras.models.load_model(weights_path + '/' + name_model + '.keras')
    print("Loaded complete model from disk")

    # Load the audio file into numpy array
    y, sr = librosa.load(audio_file, sr=sample_rate)
    audio = audio_files_to_numpy(y, None, sr, frame_length, hop_length_frame, min_duration)

    # Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    print(dim_square_spec)

    # Create Amplitude and phase of the sounds
    m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, n_fft, hop_length_fft)

    # Global scaling to have distribution between -1 and 1
    X_in = scaled_in(m_amp_db_audio)

    # Reshape for prediction (adding an extra dimension for channels)
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)

    # Prediction using the loaded model
    X_pred = loaded_model.predict(X_in)

    # Rescale back the noise model to original scale
    inv_sca_X_pred = inv_scaled_ou(X_pred)

    # Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]

    # Reconstruct audio from denoised spectrogram and phase
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)

    # Number of frames
    nb_samples = audio_denoise_recons.shape[0]

    # Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length) * 10

    # Save the noisy audio
    noisy_audio = librosa.output.write_wav('noisy_output.wav', y, sr)
    
    # Save the denoised audio to disk using soundfile.write()
    sf.write('denoised_output.wav', denoise_long[0, :], sample_rate)

    return 'noisy_output.wav', 'denoised_output.wav'

# Streamlit app
st.title('Audio Denoising App')

st.sidebar.header('Model Parameters')

# File upload widget
uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open("temp_input_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")

    # Parameters for the prediction
    sample_rate = 16000
    min_duration = 1.0
    frame_length = 2048
    hop_length_frame = 512
    n_fft = 2048
    hop_length_fft = 512

    # Run prediction
    weights_path = "path_to_weights"  # Change this path to where your model weights are stored
    name_model = "your_model_name"    # Name of your model file (without extension)
    noisy_file, denoised_file = prediction(weights_path, name_model, "temp_input_audio.wav", sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft)

    # Provide download links for the noisy and denoised audio files
    st.subheader("Noisy Audio")
    st.audio(noisy_file, format="audio/wav")
    st.download_button(label="Download Noisy Audio", data=open(noisy_file, "rb").read(), file_name="noisy_audio.wav", mime="audio/wav")

    st.subheader("Denoised Audio")
    st.audio(denoised_file, format="audio/wav")
    st.download_button(label="Download Denoised Audio", data=open(denoised_file, "rb").read(), file_name="denoised_audio.wav", mime="audio/wav")
