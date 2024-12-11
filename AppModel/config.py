# Data config =======

UPLOAD_FOLDER   = './uploads/'
SUPPORT_FORMAT  = ['mp3', 'mp4', 'wav']

NOISE_DOMAINS   = ['vacuum_cleaner', 'clapping', 'fireworks', 'door_wood_knock', 'engine', 'mouse_click', 
                    'clock_alarm', 'wind', 'keyboard_typing', 'footsteps', 'car_horn', 'drinking_sipping', 'snoring', 
                    'breathing', 'toilet_flush', 'clock_tick', 'washing_machine', 'rain', 'rooster', 'laughing']


# Speech config =====
SAMPLE_RATE         = 8000
N_FFT               = 255
HOP_LENGTH_FFT      = 63
HOP_LENGTH_FRAME    = 8064
FRAME_LENGTH        = 8064
MIN_DURATION        = 1.0