# import librosa
# import tensorflow as tf
# from tensorflow.keras.models import model_from_json
# from data_tools import scaled_in, inv_scaled_ou
# from data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio


# def prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
# audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft):
#     """ This function takes as input pretrained weights, noisy voice sound to denoise, predict
#     the denoise sound and save it to disk.
#     """

#     # load json and create model
#     json_file = open(weights_path+'/'+name_model+'.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights(weights_path+'/'+name_model+'.h5')
#     print("Loaded model from disk")

#     # Extracting noise and voice from folder and convert to numpy
#     audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate,
#                                  frame_length, hop_length_frame, min_duration)

#     #Dimensions of squared spectrogram
#     dim_square_spec = int(n_fft / 2) + 1
#     print(dim_square_spec)

#     # Create Amplitude and phase of the sounds
#     m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(
#         audio, dim_square_spec, n_fft, hop_length_fft)

#     #global scaling to have distribution -1/1
#     X_in = scaled_in(m_amp_db_audio)
#     #Reshape for prediction
#     X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
#     #Prediction using loaded network
#     X_pred = loaded_model.predict(X_in)
#     #Rescale back the noise model
#     inv_sca_X_pred = inv_scaled_ou(X_pred)
#     #Remove noise model from noisy speech
#     X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
#     #Reconstruct audio from denoised spectrogram and phase
#     print(X_denoise.shape)
#     print(m_pha_audio.shape)
#     print(frame_length)
#     print(hop_length_fft)
#     audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
#     #Number of frames
#     nb_samples = audio_denoise_recons.shape[0]
#     #Save all frames in one file
#     denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length)*10
#     librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], sample_rate)


import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from data_tools import scaled_in, inv_scaled_ou
from data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio


def prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
               audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft, use_complete_model=False):
    """ This function takes as input pretrained weights, noisy voice sound to denoise, predicts
    the denoised sound and saves it to disk.
    
    Parameters:
    - use_complete_model (bool): Whether to load the entire model (architecture + weights) from a single file or to load the model architecture from a .json file and weights from a .h5 file.
    """
    
    if use_complete_model:
        # Load the entire model (architecture + weights) from a single file
        loaded_model = tf.keras.models.load_model(weights_path + '/' + name_model + '.keras')
        print("Loaded complete model from disk")
    else:
        # Load json and create model
        json_file = open(weights_path + '/' + name_model + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        # Create model from json
        loaded_model = model_from_json(loaded_model_json)
        
        # Load weights into the model
        loaded_model.load_weights(weights_path + '/' + name_model + '.h5')
        print("Loaded model from disk")

    # Extracting noise and voice from folder and convert to numpy
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate,
                                 frame_length, hop_length_frame, min_duration)

    # Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    print(dim_square_spec)

    # Create Amplitude and phase of the sounds
    m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, n_fft, hop_length_fft)

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
    print(X_denoise.shape)
    print(m_pha_audio.shape)
    print(frame_length)
    print(hop_length_fft)

    # Convert the denoised spectrogram back to audio
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)

    # Number of frames
    nb_samples = audio_denoise_recons.shape[0]

    # Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length) * 10

    # Save the denoised audio to disk
    librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], sample_rate)
