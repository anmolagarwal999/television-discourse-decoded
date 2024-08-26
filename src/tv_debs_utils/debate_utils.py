import time
import os
import librosa
from pytubefix import YouTube
import torch
import soundfile as sf
from pydub import AudioSegment
import logging
from ..config_constants import ConfigConstants

# Constants for audio processing
sr = 16000  # sample rate
frame_dur = 0.025  # 25 ms frame size
hop_dur = 0.01  # 10 ms hop size​
sample_dur = 1  # 1 second samples
frames_per_sample = int(sample_dur / hop_dur)
mfcc_ct = 26

def get_logger():
    """
    Create and configure a logger object.

    Returns:
        logging.Logger: Configured logger object.
    """
    timestamp_epoch = int(time.time())
    logger = logging.getLogger(__name__+str(timestamp_epoch))
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger

# Get a logger to use
logger = get_logger()

def force_cudnn_initialization():
    """
    Force CUDA initialization by performing a dummy operation.
    """
    logger.debug("Force Cuda initialization")
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(
        s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def download_ytvid_as_wav(video_id: str) -> bool:
    """
    Download a YouTube video as a WAV file.

    Args:
        video_id (str): YouTube video ID.

    Returns:
        bool: True if download was successful (or if video was already present), False otherwise.
    """
    logger.info(f"video with id: {video_id} download started.")
    expected_download_path = os.path.join(
        ConfigConstants.PART_0_PATH, f"{video_id}.wav")
    mp3_path = os.path.join(ConfigConstants.MP3_FILE_DIR, f"{video_id}.mp3")
    mp4_path = os.path.join(ConfigConstants.MP3_FILE_DIR, f"{video_id}.mp3")

    # Check if the file already exists
    if os.path.exists(expected_download_path):
        logger.debug(f"Wav file for Video with id: {video_id=} already exists.")
        return True

    # Constructing the YouTube video URL using the provided video ID
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        # Creating a YouTube object by passing the video URL
        yt = YouTube(video_url)
        result = yt.streams.filter(adaptive=True, only_audio=True).first()

        # Checking if the video has all its fragments available
        if result is None:
            raise ValueError("All video fragments are not available.")

        # Download and convert the video
        out_file = result.download(
            output_path=ConfigConstants.MP3_FILE_DIR, filename=f'{video_id}.mp4')
        original_extension = out_file.split('.')[-1]
        mp3_converted_file = AudioSegment.from_file(
            out_file, original_extension)
        mp3_converted_file.export(
            expected_download_path, format='wav', bitrate="192k")

        # Clean up temporary files
        for file_path in [mp3_path, out_file, mp4_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

        logger.info(f"Video with id: {video_id} download done.")
        return True
    except Exception as e:
        logger.exception(f"Error occurred while downloading the video: {e}")
        return False

def extract_speech_segments(speech_segments):
    """
    Extract speech segments from the given speech_segments object.

    Args:
        speech_segments: Object containing speech segments.

    Returns:
        list: List of tuples containing start and end times of speech segments.
    """
    speech_present_arr = []
    for curr_elem in speech_segments:
        use_dict = curr_elem.__dict__
        use_tuple = [use_dict['start'], use_dict['end']]
        speech_present_arr.append(use_tuple)
    return sorted(speech_present_arr)

def remove_non_speech(curr_yt_id, timestamps):
    """
    Remove non-speech segments from the audio file.

    Args:
        curr_yt_id (str): YouTube video ID.
        timestamps (list): List of tuples containing start and end times of speech segments.

    Returns:
        tuple: Concatenated audio array and sample rate.
    """
    part_0_path = os.path.join(ConfigConstants.PART_0_PATH, f"{curr_yt_id}.wav")
    part_1_path = os.path.join(ConfigConstants.PART_1_PATH, f"{curr_yt_id}.wav")
    audio, sr = librosa.load(part_0_path, sr=None)  # Load the WAV file

    extracted_segments = []
    for start, end in timestamps:
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        segment = audio[start_frame:end_frame]
        extracted_segments.extend(segment)

    concatenated_audio = extracted_segments  # Concatenate the segments
    sf.write(part_1_path, concatenated_audio, sr)

    return concatenated_audio, sr

def remove_overlap(OSD, curr_yt_id):
    """
    Remove overlapping segments from the audio file.

    Args:
        OSD: Overlap Speech Detection object.
        curr_yt_id (str): YouTube video ID.

    Returns:
        list: List of non-overlapping timestamps.
    """
    video_input_path = os.path.join(ConfigConstants.PART_1_PATH, f"{curr_yt_id}.wav")

    OSD_output = OSD(video_input_path)

    overlap_timestamps_arr = []
    for curr_elem in OSD_output.__dict__['_tracks']:
        lb, ub = curr_elem.__dict__['start'], curr_elem.__dict__['end']
        overlap_timestamps_arr.append([lb, ub])

    return overlap_timestamps_arr

def write_non_overlap(curr_yt_id, timestamps):
    """
    Write non-overlapping segments to a new audio file.

    Args:
        curr_yt_id (str): YouTube video ID.
        timestamps (list): List of non-overlapping timestamps.
    """
    part_1_path = os.path.join(ConfigConstants.PART_1_PATH, f"{curr_yt_id}.wav")
    part_2_path = os.path.join(ConfigConstants.PART_2_PATH, f"{curr_yt_id}.wav")
    y, sr = librosa.load(part_1_path)

    overlap_removed_sample = []
    position_start = 0
    for curr_bounds in timestamps:
        sample_start, sample_end = curr_bounds

        # The starting position of the sample
        position_end = int(sample_start * sr)
        overlap_removed_sample.extend(y[position_start:position_end])
        # The end of the sample
        position_start = int(sample_end * sr)

    overlap_removed_sample.extend(y[position_start:])
    logger.debug(f"Writing at: {part_2_path}")
    sf.write(part_2_path, overlap_removed_sample, sr)