import os

class ConfigConstants:
    HUGGINGFACE_TOKEN = os.environ.get(
        'HUGGINGFACE_TOKEN', os.environ.get('HUGGINGFACE_TOKEN', None))

    PERSPECTIVE_API_KEYS = [os.environ.get('PERSPECTIVE_API_KEY', None)]

    # Directory paths
    ProjectDir = os.path.join(os.path.dirname(__file__), '../')

    SAVE_RESULTS_BASE_DIR = os.path.join(ProjectDir, 'data/results/')
    SCRATCH_FOLDER_DIR = os.path.join(ProjectDir, 'data/scratch_folder/')

    OSD_FILE_DIR = os.path.join(SAVE_RESULTS_BASE_DIR, "osd_data")
    VAD_FILE_DIR = os.path.join(SAVE_RESULTS_BASE_DIR, "vad_data")
    DIARIZATION_FILE_DIR = os.path.join(
        SAVE_RESULTS_BASE_DIR, "diarization_data")
    TRANSCRIPT_FILE_DIR = os.path.join(
        SAVE_RESULTS_BASE_DIR, "transcription_data")
    PERSPECTIVE_FILE_DIR = os.path.join(
        SAVE_RESULTS_BASE_DIR, "perspective_data")

    PART_0_PATH = os.path.join(SCRATCH_FOLDER_DIR, "part_0")
    PART_1_PATH = os.path.join(SCRATCH_FOLDER_DIR, "part_1")
    PART_2_PATH = os.path.join(SCRATCH_FOLDER_DIR, "part_2")
    MP3_FILE_DIR = os.path.join(SCRATCH_FOLDER_DIR, "temp_mp3_files")
    MP4_FILE_DIR = os.path.join(SCRATCH_FOLDER_DIR, "temp_mp4_files")
    UTTERANCES_FILE_DIR_TMP = os.path.join(
        SCRATCH_FOLDER_DIR, "utterances_tmp")

    all_directories = [SAVE_RESULTS_BASE_DIR, SCRATCH_FOLDER_DIR, OSD_FILE_DIR, VAD_FILE_DIR, DIARIZATION_FILE_DIR,
                       PART_0_PATH, PART_1_PATH, PART_2_PATH, MP3_FILE_DIR, UTTERANCES_FILE_DIR_TMP, TRANSCRIPT_FILE_DIR, PERSPECTIVE_FILE_DIR]

    # create the directories if they don't exist
    for _dir in all_directories:
        if not os.path.exists(_dir):
            os.makedirs(_dir, exist_ok=True)
