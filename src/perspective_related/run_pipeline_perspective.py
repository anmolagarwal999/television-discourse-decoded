from googleapiclient import discovery
import json
import os
from time import sleep
from multiprocessing import Process
from ..config_constants import ConfigConstants
from ..tv_debs_utils import debate_utils

# get a logger to use
logger = debate_utils.get_logger()
logger.info("INFO IS FROM THE LOGGER/")


def process_target(API_KEY, files):
    logger.info("Going to initialize PerspectiveAPI client.")
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    logger.debug("Client has been initialized")
    for file_id, file in enumerate(files):
        logger.info(f"Processing: [{file_id}/{len(files)}]: {file}")
        with open(os.path.join(ConfigConstants.TRANSCRIPT_FILE_DIR, f"{file}.json"), 'r') as f:
            transcript_data = json.load(f)
        write_path = os.path.join(
            ConfigConstants.PERSPECTIVE_FILE_DIR, f"{file}.json")
        if os.path.exists(write_path):
            logger.debug(f"Perspective data already exists for: {file}")
            continue
        logger.debug(f"Number of utterances is: {len(transcript_data)}")
        for ind, utterance in enumerate(transcript_data):

            logger.debug(f"Processing utterance: {ind}/{len(transcript_data)}")
            if "perspective" in utterance:
                continue
            if len(utterance["text"]):
                analyze_request = {
                    'comment': {'text': utterance['text']},
                    'requestedAttributes': {
                        'TOXICITY': {},
                        'SEVERE_TOXICITY': {},
                        'IDENTITY_ATTACK': {},
                        'THREAT': {},
                        'INSULT': {},
                        'PROFANITY': {}
                    },
                    'spanAnnotations': True,
                    'languages': ["en"]
                }
                try:
                    response = client.comments().analyze(body=analyze_request).execute()
                    transcript_data[ind]["perspective"] = response
                except Exception as e:
                    logger.exception(f"Error occurred for: {file}, {ind}: {e}")

                sleep(1.2)
            else:
                transcript_data[ind]["perspective"] = {}
        with open(write_path, 'w') as f:
            json.dump(transcript_data, f, indent=1)
        sleep(0.5)


files = os.listdir(ConfigConstants.TRANSCRIPT_FILE_DIR)
logger.info("Number of files with transcripts: ", len(files))

files_done = os.listdir(ConfigConstants.PERSPECTIVE_FILE_DIR)


files = sorted(list(set(files).difference(set(files_done))))
logger.info(
    "Number of files after removing files that are already processed: ", len(files))


files = list(filter(lambda x: "json" in x, files))
logger.info("Number of files after restricting to json files only: ", len(files))

process_files = {id: []
                 for id in range(len(ConfigConstants.PERSPECTIVE_API_KEYS))}
[process_files[idx % len(ConfigConstants.PERSPECTIVE_API_KEYS)].append(
    file_name.split('.json')[0]) for idx, file_name in enumerate(files)]


def run():
    procs = []

    for pID, API_KEY in enumerate(ConfigConstants.PERSPECTIVE_API_KEYS):
        proc = Process(target=process_target, args=(
            API_KEY, process_files[pID], ))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


run()
