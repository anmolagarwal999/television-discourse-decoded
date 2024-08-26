import os
import sys
import json
import glob
from pydub import AudioSegment
import whisper
from ..config_constants import ConfigConstants
from ..tv_debs_utils import debate_utils

# Set environment variable for Hugging Face model cache
# os.environ['HF_HOME'] = 'mounted_dump/hf_model_cache'

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

# Load Whisper model
whisper_model = whisper.load_model("large-v2", download_root=os.environ['HF_HOME'])

def process_video(curr_yt_id, curr_vid_idx):
    """
    Process a single video through the entire pipeline.

    Args:
        curr_yt_id (str): YouTube video ID
        curr_vid_idx (int): Index of the current video in the list

    Returns:
        bool: True if processing was successful, False otherwise
    """
    part_0_path = os.path.join(ConfigConstants.PART_0_PATH, f"{curr_yt_id}.wav")
    part_1_path = os.path.join(ConfigConstants.PART_1_PATH, f"{curr_yt_id}.wav")
    part_2_path = os.path.join(ConfigConstants.PART_2_PATH, f"{curr_yt_id}.wav")

    # Check if diarization data exists
    diarization_file_path = os.path.join(ConfigConstants.DIARIZATION_FILE_DIR, f"{curr_yt_id}.json")
    if not os.path.exists(diarization_file_path):
        logger.debug(f"Diarization data doesn't exist for: {curr_yt_id} | hence, skipping it currently.")
        return False

    # Check if transcription data already exists
    transcription_file_path = os.path.join(ConfigConstants.TRANSCRIPT_FILE_DIR, f"{curr_yt_id}.json")
    if os.path.exists(transcription_file_path):
        logger.debug(f"Transcript data already exists for: {curr_yt_id}")
        return True

    # Download the video
    if not debate_utils.download_ytvid_as_wav(curr_yt_id):
        logger.debug(f"Video download failed for {curr_yt_id=}")
        return False

    # Load VAD data and remove non-speech
    with open(os.path.join(ConfigConstants.VAD_FILE_DIR, f"{curr_yt_id}.json"), 'r') as fd:
        vad_data = json.load(fd)
    debate_utils.remove_non_speech(curr_yt_id, vad_data)

    # Load OSD data and remove overlap
    with open(os.path.join(ConfigConstants.OSD_FILE_DIR, f"{curr_yt_id}.json"), 'r') as fd:
        osd_data = json.load(fd)
    debate_utils.write_non_overlap(curr_yt_id, osd_data)

    # Load diarization data
    with open(os.path.join(ConfigConstants.DIARIZATION_FILE_DIR, f"{curr_yt_id}.json"), 'r') as fd:
        dia_data = json.load(fd)

    # Clear temporary utterance files
    files = glob.glob(os.path.join(ConfigConstants.UTTERANCES_FILE_DIR_TMP, "*"))
    for f in files:
        os.remove(f)

    # Split audio into utterances
    audio = AudioSegment.from_wav(part_2_path)
    wav_names = []
    print(f"Processing {curr_yt_id} at position {curr_vid_idx}")
    for utter in dia_data:
        start = utter[0]['start'] * 1000  # convert to millisecond
        end = utter[0]['end'] * 1000  # convert to millisecond
        speaker_id = list(utter[1].values())[0]
        name = f'{start}-{end}-{speaker_id}'
        wav_names.append(name)
        audio[start:end].export(os.path.join(ConfigConstants.UTTERANCES_FILE_DIR_TMP, f"{name}.wav"), format='wav')

    # Transcribe utterances
    trans_data = []
    logger.debug(f"Using whisper, now starting to transcribe {curr_yt_id}")
    for i, wav_name in enumerate(wav_names):
        result = whisper_model.transcribe(os.path.join(ConfigConstants.UTTERANCES_FILE_DIR_TMP, f"{wav_name}.wav"), language="en")
        useful_data = {
            'text': result['text'],
            'language': result['language'],
            'segment_start': wav_name.split('-')[0],
            'segment_end': wav_name.split('-')[1],
            'speaker': wav_name.split('-')[2]
        }
        if 'segments' in result and len(result['segments']) > 0 and 'no_speech_prob' in result['segments'][0]:
            useful_data['no_speech_prob'] = result['segments'][0]['no_speech_prob']
        trans_data.append(useful_data)
        logger.debug(f"{i} wav transcribed out of {len(wav_names)}")

    # Save transcription data
    transcript_path = os.path.join(ConfigConstants.TRANSCRIPT_FILE_DIR, f"{curr_yt_id}.json")
    with open(transcript_path, 'w') as f:
        json.dump(trans_data, f, indent=2)
    logger.debug(f"Transcription done for {curr_yt_id}")

    # Remove intermediate files
    for path in [part_0_path, part_1_path, part_2_path]:
        os.remove(path)
    logger.debug(f"Removed intermediate data for {curr_yt_id}")

    return True

# Main processing loop
error_ids = []
for curr_vid_idx, curr_yt_id in enumerate(vid_id_list):
    try:
        logger.info(f"Starting to process: {curr_vid_idx}/{len(vid_id_list)}: {curr_yt_id}")
        process_video(curr_yt_id, curr_vid_idx)
    except Exception as e:
        logger.exception(f"Error in processing {curr_yt_id}: {e}")
        error_ids.append([curr_yt_id, f"{e}"])
        for path in [os.path.join(ConfigConstants.PART_0_PATH, f"{curr_yt_id}.wav"),
                     os.path.join(ConfigConstants.PART_1_PATH, f"{curr_yt_id}.wav"),
                     os.path.join(ConfigConstants.PART_2_PATH, f"{curr_yt_id}.wav")]:
            if os.path.exists(path):
                os.remove(path)

# Clean up temporary files
files = glob.glob(os.path.join(ConfigConstants.UTTERANCES_FILE_DIR_TMP, "*"))
for f in files:
    os.remove(f)

logger.info("ENTIRE RAN. Done")