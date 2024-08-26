import os
import sys
import json
import torch
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
from ..config_constants import ConfigConstants
from ..tv_debs_utils import debate_utils

# Utility functions for debate processing
CURR_FILE_DIR = os.path.dirname(__file__)

# Get a logger to use
logger = debate_utils.get_logger()

def load_video_ids(args_received):
    """
    Load video IDs from command line argument or JSON file.

    Args:
        args_received (str): Command line argument (video ID or path to JSON file)

    Returns:
        list: List of video IDs
    """
    if args_received.endswith(".json"):
        assert os.path.exists(args_received)
        with open(args_received) as fd:
            return json.load(fd)
    else:
        return [args_received]

# Load video IDs
vid_id_list = load_video_ids(sys.argv[1])

# Hyperparameters for Audio Processing
HYPER_PARAMETERS = {
    "onset": 0.5,  # onset activation threshold
    "offset": 0.5,  # offset activation threshold
    "min_duration_on": 0.0,  # remove speech regions shorter than this (in seconds)
    "min_duration_off": 0.0,  # fill non-speech regions shorter than this (in seconds)
}

def load_models():
    """
    Load and initialize all required models for audio processing.

    Returns:
        tuple: Containing initialized models (speaker_diarization_model, segmentation_model, VAD, OSD)
    """
    # Load speaker diarization pipeline model
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=ConfigConstants.HUGGINGFACE_TOKEN)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline.to(device)
    speaker_diarization_model = pipeline
    logger.info("Speaker diarization model loaded.")

    # Load segmentation model for VAD and OSD
    segmentation_model = Model.from_pretrained(
        "pyannote/segmentation", use_auth_token=ConfigConstants.HUGGINGFACE_TOKEN).to(device)
    logger.info("Segmentation model loaded.")

    # Load Voice Activity Detection (VAD) model
    VAD = VoiceActivityDetection(segmentation=segmentation_model)
    VAD.instantiate(HYPER_PARAMETERS)
    logger.info("VAD model loaded.")

    # Load Overlapped Speech Detection (OSD) model
    OSD = OverlappedSpeechDetection(segmentation=segmentation_model)
    OSD.instantiate(HYPER_PARAMETERS)
    logger.info("OSD model loaded.")

    return speaker_diarization_model, segmentation_model, VAD, OSD

# Load all required models
speaker_diarization_model, segmentation_model, VAD, OSD = load_models()

def process_video(curr_yt_id, VAD, OSD, speaker_diarization_model):
    """
    Process a single video through the entire pipeline.

    Args:
        curr_yt_id (str): YouTube video ID
        VAD: Voice Activity Detection model
        OSD: Overlapped Speech Detection model
        speaker_diarization_model: Speaker diarization model

    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Check if diarization data already exists
    diarization_file_path = os.path.join(
        ConfigConstants.DIARIZATION_FILE_DIR, f"{curr_yt_id}.json")
    if os.path.exists(diarization_file_path):
        logger.debug(f"Diarization data already exists for: {curr_yt_id}")
        return True

    # Step 1: Download the video and save as WAV
    if not debate_utils.download_ytvid_as_wav(curr_yt_id):
        logger.debug(f"Video download failed for {curr_yt_id=}")
        return False

    # Step 2: Apply VAD on the video
    part_0_path = os.path.join(ConfigConstants.PART_0_PATH, f"{curr_yt_id}.wav")
    output = VAD(part_0_path)
    logger.debug("VAD model applied")
    speech_segments = output.get_timeline().support()
    ans = debate_utils.extract_speech_segments(speech_segments)

    save_path = os.path.join(ConfigConstants.VAD_FILE_DIR, f"{curr_yt_id}.json")
    with open(save_path, 'w') as fd:
        json.dump(ans, fd, indent=1)

    # Remove non-speech areas and save the WAV
    debate_utils.remove_non_speech(curr_yt_id, ans)

    # Step 3: Apply OSD on the video
    ans_2 = debate_utils.remove_overlap(OSD, curr_yt_id)
    save_path = os.path.join(ConfigConstants.OSD_FILE_DIR, f"{curr_yt_id}.json")
    with open(save_path, 'w') as fd:
        json.dump(ans_2, fd, indent=1)
    debate_utils.write_non_overlap(curr_yt_id, ans_2)

    # Step 4: Get diarization data
    save_path = os.path.join(ConfigConstants.DIARIZATION_FILE_DIR, f"{curr_yt_id}.json")
    part_2_path = os.path.join(ConfigConstants.PART_2_PATH, f"{curr_yt_id}.wav")
    dz = speaker_diarization_model(part_2_path)
    logger.debug("Diarization running done")
    dia_ans = dict(dz.__dict__['_tracks']).items()
    dia_ans = [(x[0].__dict__, x[1]) for x in dia_ans]
    with open(save_path, 'w') as fd:
        json.dump(dia_ans, fd, indent=1)

    # Step 5: Clean up intermediate files
    for path in [part_0_path, os.path.join(ConfigConstants.PART_1_PATH, f"{curr_yt_id}.wav"), part_2_path]:
        os.remove(path)
    logger.debug("Removed intermediate data")

    return True

for curr_vid_idx, curr_yt_id in enumerate(vid_id_list):
    logger.debug(f"Starting to process: {curr_vid_idx}/{len(vid_id_list)}: {curr_yt_id}")
    process_video(curr_yt_id, VAD, OSD, speaker_diarization_model)

logger.debug("ENTIRE PROCESS COMPLETED. Done")