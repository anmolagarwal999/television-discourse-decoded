# Television Discourse Decoded: Comprehensive Multimodal Analytics at Scale

This repository contains the official code and data for [our paper](https://dl.acm.org/doi/10.1145/3637528.3671532) **"Television Discourse Decoded: Comprehensive Multimodal Analytics at Scale"**, accepted at **KDD'2024**.

Our work introduces an automated toolkit that leverages state-of-the-art computer vision and speech-to-text techniques to transcribe, diarize, and analyze thousands of YouTube videos from televised debates, offering profound insights into biases, incivility, and the overall quality of public discourse.


## Repository Structure and Contents

### Code
* **`config_constants.py`**: Contains configurable parameters, including API keys (such as Hugging Face tokens, Perspective API keys), and paths for storing intermediate data.
* **`diarization_vad_osd_related/run_pipeline_osd_vad.py`**: Processes a video by removing segments where no voice activity is detected using Voice Activity Detection (VAD).
* **`perspective_related/run_pipeline_perspective.py`**: Analyzes the foul speech content for each utterance, assessing it across various dimensions (e.g., identity attack, profanity) based on the spoken content.
* **`transcription_related/run_pipeline_transcription.py`**: Transcribes each utterance identified during the diarization process, providing a text representation of the spoken content.
* **`tv_debs_utils/debate_utils.py`**: Contains a set of utility functions for processing, downloading, and truncating videos.

### Dataset
This dataset contains metadata and labels for YouTube videos used in our work. Each entry in the dataset corresponds to a video and includes various fields detailing the video's attributes, statistics, and detected hashtags.
- **`video_idx`**: A unique index assigned to each video in the dataset.
- **`yt_vid_id`**: The unique identifier for the video on YouTube. This is the `videoId` that appears in YouTube URLs.
- **`yt_vid_url`**: The full URL to the video on YouTube.
- **`major_label`**: The primary category or theme associated with the video. This provides a high-level categorization of the video's content.
- **`minor_labels`**: A list of secondary labels or subcategories that further describe the video's content. These labels offer more granular categorization.
- **`yt_stats`**: A dictionary containing statistics related to the video on YouTube.
- **`publish_time`**: The timestamp indicating when the video was published on YouTube.
- **`vid_title`**: The title of the video as it appears on YouTube.
- **`total_duration`**: The total duration of the video in seconds.
- **`total_duration_str`**: The total duration of the video in ISO 8601 duration format.
- **`hashtags_detected`**: A list of hashtags detected in the video's description or title or via OCR on the video frames.

### Results and Intermediate files related
* **`data/scratch_folder`**
    * **`./part_0`**: Downloads the video from YouTube using the YT ID.
    * **`./part_1`**: Processes the video from `part_0`; contains the audio versions of the debates after applying Voice Activity Detection (VAD), removing segments where no voice was detected.
    * **`./part_2`**: Processes the audio from `part_1` by removing segments where more than one speaker is detected, using Overlapped Speech Detection (OSD).

* **`data/results`**
    * **`./osd_data`**: Stores timestamps for segments of the video where multiple speakers were detected, indicating overlapping speech.
    * **`./vad_data`**: Stores timestamps for segments of the video where any voice activity was detected.
    * **`./diarization_data`**: Contains timestamps for segments of the video where different speakers were detected. Includes speaker IDs, maintaining consistent identification for each speaker throughout the video, numbered from 0 to N-1, where N is the total number of speakers.
    * **`./transcription_data`**: Provides detailed information about each utterance, including the content of the speech, timestamps of the utterance, and the speaker ID associated with it.
    * **`./perspective_data`**: Contains information on any foul language or offensive content found in the transcript, with details linked to specific utterances.


## Getting Started

### Setting up the Project Repository

To get started with the project, follow these steps:

1. Clone the repository to your local machine:
```bash
git clone https://github.com/anmolagarwal999/television-discourse-decoded
cd television-discourse-decoded
```

2. Create and activate the conda environment:
```bash
conda create --name tv_debs_env python=3.9
conda activate tv_debs_env
```
3. Install the required dependencies
```bash
pip install -r requirements.txt
```

## Citation
Please consider citing the following paper when using our code and dataset.

```
@inproceedings{10.1145/3637528.3671532,
author = {Agarwal, Anmol and Priyadarshi, Pratyush and Sinha, Shiven and Gupta, Shrey and Jangra, Hitkul and Kumaraguru, Ponnurangam and Garimella, Kiran},
title = {Television Discourse Decoded: Comprehensive Multimodal Analytics at Scale},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671532},
doi = {10.1145/3637528.3671532},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {4752–4763},
numpages = {12},
keywords = {bias detection, incivil speech, multimodal analysis, television, video analysis},
location = {Barcelona, Spain},
series = {KDD '24}
}
```