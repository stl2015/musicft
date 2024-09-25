"""
Script for fetching instruction-tuning data from Mixtral.

Usage:

# jamendo mir
python scripts/fetch_openai_instruct_data.py \
    --data-source datasets/mtg-jamendo/jamendo-annotated.jsonl \
    --dataset-name mtg-jamendo \
    --prompt-type mir \
    --runner DataflowRunner


"""
import argparse
import datetime
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union
from dataclasses import dataclass
from transformers import AutoTokenizer
import transformers
import torch
import glob
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod

import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import storage

# set the below to match your setup
DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
NUM_RETRIES = 1
GCP_PROJECT_NAME = ""
GCS_BUCKET_NAME = ""
US_CENTRAL1_REGION = "us-central1"

PROMPT = ""

EXPECTED_FIELDS = [
    "context_activities",
    "context_cultural",
    "genre",
    "mood",
    "sound_descriptions",
    "music_descriptions",
    "music_analysis",
    "music_creation",
    "abstract",
]

OPTIONAL_FIELDS = ["language", "lyrics", "vocals", "instruments", "rhythm"]
ALLOWED_FIELDS = set(["title", "artist", "uri"] + EXPECTED_FIELDS + OPTIONAL_FIELDS)


import random
from typing import Any, Dict, Sequence

LONG_CAPTION_PROMPTS = [
    "Describe the song in detail.",
    "Provide an elaborate description of the song.",
    "Break down the song with great detail.",
    "Offer a thorough explanation of the musical piece.",
    "Present an intricate account of the song that follows.",
    "Describe the details of what you hear in the musical composition.",
    "Describe the musical piece comprehensively.",
    "Analyze the intricacies of the musical audio.",
    "Paint a detailed picture of the song.",
    "Unravel the intricacies of the musical piece.",
    "Examine the audio closely and share its details.",
    "What does this music sound like? Give a detailed description.",
    "What happens in this song? Present a thorough analysis.",
    "Examine the song with meticulous attention to detail.",
    "Narrate the contents of the audio with precision.",
]

# Prompts for short, informal descriptions,
# e.g. those in MusicCaps or YT8M-MusicTextClips.
SHORT_CAPTION_PROMPTS = [
    "Give a short, informal summary of the clip.",
    "What does this music sound like? Give a short description.",
    "Provide an overview of the musical content of the clip.",
    "What does this music sound like?",
    "Briefly narrate the contents of the provided music.",
    "How would you summarize this song?",
    "What happens in this song? Provide a brief summary.",
    "Please provide a quick summary of the musical audio.",
    "Explain the contents of this song.",
    "What do you hear in this music? Give a short summary.",
    "Please provide a cursory description of the song.",
    "Give a short description of this music.",
    "What is happening in this clip? Provide a brief description.",
    "Give a short synopsis of the provided music.",
    "How would you briefly caption this audio?",
    "Offer a concise portrayal of the audio clip's musical essence.",
    "Present a succinct breakdown of the musical composition in the clip.",
    "Describe the auditory characteristics of this music in a few words.",
    "Summarize the key elements found within this musical piece.",
    "Outline the main features of the provided music in brief.",
    "Provide a succinct account of the sonic content in the clip.",
    "Give a quick rundown of what this musical excerpt entails.",
    "Elaborate briefly on the musical components present in the recording.",
    "Sum up your perception of the music in a concise manner.",
    "Deliver a short, descriptive overview of the song's auditory elements.",
    "Summarize the musical content of the audio.",
    "Give a short and clear description of the clip provided.",
    "What do you hear in the provided music excerpt?",
]

# Mappping of dataset names to caption prompts. We use 'long' captions
# for datasets with note- and instrument-level information.
CAPTIONING_PROMPTS = {
    "musiccaps": SHORT_CAPTION_PROMPTS,
    "yt8m-musictextclips": SHORT_CAPTION_PROMPTS,
    "musicnet": LONG_CAPTION_PROMPTS,
    "slakh": LONG_CAPTION_PROMPTS,
    "fsl10k": SHORT_CAPTION_PROMPTS,
}


def is_caption_resonse(elem) -> bool:
    return "caption" in elem["response"]


def insert_caption_qa(elem: Dict[str, Any], caption_prompts: Sequence[str]) -> Dict[str, Any]:
    """Randomly select a prompt from caption_prompts and insert it."""
    caption_prompt = random.choice(caption_prompts)
    caption = elem["response"]["caption"]
    elem["response"] = [{"question": caption_prompt, "answer": caption}]
    return elem


@dataclass
class DatasetInfo:
    """Class to represent information about a dataset.

    By default, datasets have a unique identifying field called 'id',
    and this field is used to fetch the audio via {id}.wav. If this is
    *not* true for a dataset, then the method .id_to_filename() and id_col
    may need to be overriden.
    """

    id_col: str = "id"
    caption_col: Optional[str] = None

    def preprocess_id_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function to apply any preprocessing to the id column."""
        df[self.id_col] = df[self.id_col].astype(str)
        return df

    def id_to_filename(self, track_id: str, dirname: Optional[str] = None):
        if not isinstance(track_id, str):
            track_id = str(track_id)
        filename = str(track_id) + ".wav"
        if dirname:
            filename = os.path.join(dirname, filename)
        return filename

    @property
    def caption_prompts(self) -> Union[List[str], None]:
        return CAPTIONING_PROMPTS.get(self.name)

@dataclass
class MusicNetDatasetInfo(DatasetInfo):
    name = "musicnet"

DATASET_INFO: Dict[str, DatasetInfo] = {
    "musicnet": MusicNetDatasetInfo(),
}

def read_jsonl_data(path: str) -> pd.DataFrame:
    """Read JSONL file(s) from a wildcard path and return a DataFrame."""
    files = glob.glob(path)
    if not len(files):
        raise ValueError(f"no files found matching {path}")
    out = []
    for f in tqdm(files, desc=f"read {path}"):
        annotations = pd.read_json(path_or_buf=f, lines=True)
        out.append(annotations)

    if len(out) > 1 and not all(
        set(out[0].columns) == set(out[j].columns) for j in range(1, len(out))
    ):
        logging.warning(
            "got different sets of columns for different datasets;"
            " there may be an alignment issue with the data."
        )

    df = pd.concat(out)
    
    return df

class LLMsArentPerfectAtGeneratingJSON(ValueError):
    pass

@dataclass
class FewShotExample:
    user: Any
    assistant: List[Dict[str, str]]

def oxford_comma(x: List[str]) -> str:
    if len(x) == 1:
        return x[0]
    elif len(x) == 2:
        return " and ".join(x)
    else:
        return ", ".join(x[:-1]) + ", and " + x[-1]

def parse_almost_json(response: str):
    """
    Parse a JSON object or array that should be valid, but might be missing a brace
    or bracket here or there.

    This is used when we're asking a Large Language Model to generate syntactically
    valid JSON for us. This alone is a sign that we're living in the future, but alas,
    the future still has some problems we need to deal with.

    Sometimes, the LLM misses the mark a bit and forgets to close a brace on the end,
    of a JSON object,  or adds an extra character (or three) on the end. This function
    attempts to parse the provided JSON string a bit more tolerantly.
    """
    for suffix in ("", "]", "}", "}]"):
        try:
            return json.loads(response + suffix)
        except Exception as e:
            if "extra data" in str(e):
                limit = int(str(e).split("char ")[1].split(")")[0])
                try:
                    return json.loads(response[:limit])
                except Exception:
                    pass

    # If none of the above attempts at parsing worked, try cutting the end of the string:
    for to_cut in range(0, 100):
        try:
            return json.loads(response[:-to_cut])
        except Exception:
            pass

    # If none of _those_ attempts worked, well, throw something:
    raise LLMsArentPerfectAtGeneratingJSON(
        f"OpenAI returned a JSON response that was not syntactically valid: {response!r}"
    )

def unnest_list(list_in):
    # recursive unnesting / ignoring nested dictionaries
    def _unnest(a_list):
        for e in a_list:
            if isinstance(e, list):
                _unnest(e)  # recurse if list
            elif isinstance(e, dict):
                pass  # don't know how to handle nested dictionaries, ignore
            else:
                yield e

    return list(_unnest(list_in))


def correct_element(input_row: Dict) -> Dict:
    """
    Apply a series of corrections to the input dictionary, to
    constrain GPT's "creativity":
    - no nested arrays (e.g.: {"languages": ["de","en",[]]} -- if
      present, flatten them)
    - check that the values in the dictionary are lists of individual
      elements (i.e., that returned values don't contain list of
      dictionaries -- in that case, ignore those dictionaries)
    - if a field (aside from uri/title/artist) is a string, make it a [string]
    - the language field is not null (rather an empty list) -- because
      the schema auto-detection guesses that language is NOT an
      optional field
    - no other fields than the ones requested (i.e., that gpt didn't invent
      a field)
    """
    output_row = {}
    # break nested return values (e.g.: "languages": ["de","en",[]]) and set
    for key, value_in in input_row.items():
        output_row[key] = unnest_list(value_in) if isinstance(value_in, list) else value_in
    # make sure each openai field is a list
    for key in EXPECTED_FIELDS + OPTIONAL_FIELDS:
        if key in output_row:
            if isinstance(output_row[key], str):
                output_row[key] = [output_row[key]]
    # make sure the language field is not null
    if output_row.get("language") is None:
        output_row["language"] = []
    # make sure there are no invented fields
    output_row = {key: value for key, value in output_row.items() if key in ALLOWED_FIELDS}
    return output_row


@dataclass
class PromptHelper(ABC):
    few_shot: bool
    prompt_text: str
    few_shot_examples: Optional[Sequence[FewShotExample]] = None

    def get_prompt_text(self) -> str:
        """Fetch the prompt text."""
        return self.prompt_text

    @abstractmethod
    def get_chatgpt_query(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the query text to provide to ChatGPT."""
        raise

    def build_messages(self, prompt_text, query) -> List[Dict[str, str]]:
        """Builds the `messages` attribute to use for openai.ChatCompletion.create()."""
        fewshot_examples_formatted = []
        if self.few_shot:
            for fewshot_example in self.few_shot_examples:
                fewshot_examples_formatted.append(
                    {
                        "role": "user",
                        "content": json.dumps(fewshot_example.user),
                    }
                )
                fewshot_examples_formatted.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(fewshot_example.assistant),
                    }
                )
        return [
            # {"role": "system", "content": prompt_text},
            # *fewshot_examples_formatted,
            {"role": "user", "content": prompt_text+json.dumps([query])},
        ]

    @abstractmethod
    def postprocess_response_text(self, text: str, query, uri) -> Dict[str, Any]:
        """Postprocess the ChatGPT response and return a (possibly cleaned) version."""
        raise

    @abstractmethod
    def check_chatgpt_response_meets_schema(
        self, response: Union[Dict, List[Dict]]
    ) -> Union[Dict, List[Dict]]:
        raise

@dataclass
class BasicPromptHelper(PromptHelper):
    """Helper for the default prompt type."""

    few_shot = False
    prompt_text = PROMPT

    def get_chatgpt_query(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the query text to provide to ChatGPT."""
        track = metadata["name"]
        artists = oxford_comma([a["name"] for a in metadata["artist"]])
        return {"title": track, "artist": artists}

    def postprocess_response_text(self, text: str, query, uri) -> Dict[str, Any]:
        """Postprocess the ChatGPT response and return a (possibly cleaned) version."""
        response = self.check_chatgpt_response_meets_schema(parse_almost_json(text)[0])
        row = dict(list(response.items()) + list(query.items()) + [("uri", uri)])
        row = correct_element(row)
        return row

    def check_chatgpt_response_meets_schema(
        self, response: Union[Dict, List[Dict]]
    ) -> Union[Dict, List[Dict]]:
        assert isinstance(response, dict)
        expected_fields = EXPECTED_FIELDS
        optional_fields = OPTIONAL_FIELDS

        for expected_field in expected_fields:
            if expected_field not in response:
                raise ValueError(f"Missing field from ChatGPT response: {expected_field}")
        for optional_field in optional_fields:
            if optional_field not in response:
                response = dict(response.items())
                response[optional_field] = []
        return response

class CaptioningPromptHelper(PromptHelper):
    def get_chatgpt_query(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the query text to provide to ChatGPT."""
        return metadata

    def check_chatgpt_response_meets_schema(
        self, response: Union[Dict, List[Dict]]
    ) -> Union[Dict, List[Dict]]:
        """A no-op; captions are text-only."""
        return response

    def postprocess_response_text(self, text: str, query, uri) -> Dict[str, Any]:
        """Postprocess the ChatGPT response and return a (possibly cleaned) version."""
        # For captioning, the output is just text.
        response = self.check_chatgpt_response_meets_schema(text)
        row = dict(list(query[0].items()) + [("uri", uri)])
        row["response"] = {"caption": response}
        return row


def get_prompt_helper(prompt_type, dataset_info: DatasetInfo, few_shot: bool) -> PromptHelper:
    # Get the prompt text.
    if prompt_type == "default":
        prompt_text = PROMPT
    else:
        prompt_file = f"{prompt_type}-{dataset_info.name}-prompt.txt"
        prompt_text = open(os.path.join(os.path.dirname(__file__), prompt_file)).read()

    # Fetch the PromptHelper class.
    if prompt_type == "default":
        if few_shot:
            logging.warning("few_shot is True but BasicPrompter is selected.")
        helper_cls = BasicPromptHelper
    elif prompt_type == "captioning":
        helper_cls = CaptioningPromptHelper
    else:
        raise NotImplementedError(f"prompt type {prompt_type} not implemented.")

    return helper_cls(few_shot=few_shot, prompt_text=prompt_text)

class StreamIntoFiles(beam.DoFn):
    """
    Write files, but don't end the pipeline - pass the written
    file paths on to the next transform.
    This transform avoids a shuffle step and avoids buffering data in memory,
    but doesn't allow control over the number of output files (num_shards).
    """

    def __init__(
        self,
        output_path: str,
        file_name_suffix=".txt",
        max_records_per_file: Optional[int] = 500,
    ):
        self.output_path = output_path
        self.file_name_suffix = file_name_suffix
        self.max_records_per_file = max_records_per_file

        if "gs://" in self.output_path:
            self.bucket_name = output_path.split("gs://")[1].split("/")[0]
            self.blob_prefix = output_path.replace(f"gs://{self.bucket_name}/", "")
            if not self.blob_prefix.endswith("/"):
                self.blob_prefix = self.blob_prefix + "/"

    def setup(self) -> None:
        self.log = logging.getLogger()
        if "gs://" in self.output_path:
            self.storage_client = storage.Client()

    def open_file(self):
        # import tensorflow.io.gfile

        self.bundle_uuid = str(uuid.uuid4()).replace("-", "")
        if "gs://" in self.output_path:
            self.bucket = self.storage_client.bucket(self.bucket_name)
            blob_name = f"{self.blob_prefix}{self.bundle_uuid}{self.file_name_suffix}"
            self.blob = self.bucket.blob(blob_name)
            self.handle = self.blob.open("wb")
            self.log.info(f"Opened: gs://{self.blob.bucket.name}/{self.blob.name}")
            self.handle.write(b"")
        else:
            self.filename = f"{self.output_path}/{self.bundle_uuid}{self.file_name_suffix}"
            # tensorflow.io.gfile.makedirs(os.path.dirname(self.filename))
            self.handle = open(self.filename, "wb")
            self.log.info(f"Opened: {self.filename}")
        self.records_written = 0

    def start_bundle(self):
        self.open_file()

    def process(self, value):
        self.write_record_to_handle(value, self.handle)
        self.records_written += 1
        if self.records_written >= self.max_records_per_file:
            self.close_file()
            self.open_file()

    def write_record_to_handle(self, record, handle):
        if isinstance(record, str):
            self.handle.write(record.encode("utf-8"))
            if not record.endswith("\n"):
                self.handle.write(b"\n")
        elif isinstance(record, bytes):
            self.handle.write(record)
        else:
            raise NotImplementedError(
                f"Not sure how to write '{type(record)}' objects "
                f"to file extension {self.file_name_suffix}!"
            )

    def close_file(self):
        self.handle.close()
        if "gs://" in self.output_path:
            path = f"gs://{self.blob.bucket.name}/{self.blob.name}"
        else:
            path = self.filename
        self.log.info(f"Closed: {path}")

    def finish_bundle(self):
        self.close_file()


def response_contains_metadata_filter(response_text) -> bool:
    return "metadata" in response_text.lower()


def prompt(model: str, metadata, prompt_helper: PromptHelper) -> Dict[str, List[str]]:
    """
    Given track metadata, ask ChatGPT to answer the prompt, and return ChatGPT's response
    as a JSON dict with the metadata added back in.
    """
    uri = [x['id'] for x in metadata]
    
    query = prompt_helper.get_chatgpt_query(metadata)
    last_exception = None
    prompt_text = prompt_helper.get_prompt_text()
    messages = prompt_helper.build_messages(prompt_text, query)

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
    )   

    # This needs to be done at runtime on each Dataflow worker:
    # TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=OPENAI_ORGANIZATION)'
    # openai.organization = OPENAI_ORGANIZATION

    for retry in range(NUM_RETRIES):
        try:

            prompt_type = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response_iterator = pipeline(prompt_type, max_new_tokens=1024, do_sample=True, temperature=0.01, top_k=50, top_p=0.95)

            response_chunks = []
            for chunk in response_iterator:
                text = chunk["generated_text"].split('[/INST]')[1]
                if text:
                    response_chunks.append(text)
            text = "".join(response_chunks)            
            return prompt_helper.postprocess_response_text(text, query, uri)
        except Exception as e:
            last_exception = e
            time.sleep(2**retry)
    print(f"Failed to fetch data for {uri} due to {last_exception}")
    return {"uri": uri, "exception": repr(last_exception)}


def main():
    parser = argparse.ArgumentParser(
        description=("Query OpenAI's model(s) for structured information about Spotify track URIs.")
    )

    parser.add_argument("--output-path", help="The output file to save JSONL data to.")
    parser.add_argument(
        "--runner",
        help="Which Beam runtime to use.",
        choices=["DataflowRunner", "DirectRunner"],
        default="DataflowRunner",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="The model to use.",
        choices=["mistralai/Mixtral-8x7B-Instruct-v0.1"],
    )
    parser.add_argument(
        "--num-threads-per-worker",
        default=4,
        type=int,
        help=(
            "The number of parallel requests to make of OpenAI per Dataflow worker. "
            "Multiplied by --num-workers, this roughly gives the total number of simultaneous "
            "requests. Turn this number down if getting errors or OpenAI rate limits."
        ),
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="The number of Dataflow workers to spin up.",
    )
    parser.add_argument(
        "--worker-disk-size-gb",
        default=128,
        type=int,
        help="Worker disk size in GB. Note that disk size must be at least size of the "
        + "docker image.",
    )

    parser.add_argument(
        "--data-source",
        required=True,
        help="Data source to use. Should be a path (or wildcard) to a valid JSONL file(s).",
    )

    parser.add_argument(
        "--dataset-name",
        choices=list(DATASET_INFO.keys()),
        required=True,
        help="Name of the dataset being used. This is required to select "
        + "the correct prompt template.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit on number of samples to try. If the provided dataset is larger, "
        + "it will be downsampled.",
    )
    parser.add_argument(
        "--few-shot",
        default=False,
        action="store_true",
        help="Whether to use few-shot prompting to GPT. "
        "If True, use few-shot examples for prompting. See PromptHelper for more info.",
    )

    parser.add_argument(
        "--prompt-type",
        default="default",
        choices=["default", "captioning"],
        help="the type of prompt to use.",
    )
    parser.add_argument("--drop-columns", default=None, nargs="+")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    date_slug = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    if args.output_path is None:
        args.output_path = f"gs://{GCS_BUCKET_NAME}/openai-data/{date_slug}/{args.model}/"

    if args.runner == "DataflowRunner":
        job_name = f"music2text-{args.model.replace('.', '-')}-scraper-{date_slug}"[:128]
        print(f"🏠 Output data will be written to: {args.output_path}")
        print(f"🚀 Starting a Dataflow job named: {job_name}...")

        pipeline_options = {
            "runner": "DataflowRunner",
            "project": GCP_PROJECT_NAME,
            "job_name": job_name,
            "region": US_CENTRAL1_REGION,
            "machine_type": "n1-highcpu-4",
            "max_num_workers": args.num_workers,
            "worker_disk_type": "pd-ssd",
            "disk_size_gb": args.worker_disk_size_gb,
            "experiments": ["use_runner_v2", "beam_fn_api"],
            # Control concurrency here to avoid DDOS'ing OpenAI:
            "number_of_worker_harness_threads": args.num_threads_per_worker,
            "save_main_session": True,
            "sdk_container_image": "gcr.io/path/to/m2t-preprocess:latest",
            "temp_location": f"gs://{GCS_BUCKET_NAME}/dataflow-tmp",
        }
    else:
        pipeline_options = {
            "runner": args.runner,
            "project": GCP_PROJECT_NAME,
            "temp_location": f"gs://{GCS_BUCKET_NAME}/dataflow-tmp",
        }

    pipeline_options = PipelineOptions(**pipeline_options)

    dataset_info = DATASET_INFO[args.dataset_name]
    uri_key = dataset_info.id_col
    prompt_helper = get_prompt_helper(args.prompt_type, dataset_info, few_shot=args.few_shot)
    with beam.Pipeline(options=pipeline_options) as p:
        df = read_jsonl_data(args.data_source)
        if args.drop_columns:
            df.drop(columns=args.drop_columns, inplace=True)

        if args.max_samples is not None and args.max_samples < len(df):
            logging.warning(f"downsampling data from size {len(df)} to {args.max_samples}")
            df = df.sample(n=args.max_samples)

        tmp = df.to_dict("records")
        res = prompt(args.model, tmp, prompt_helper)

        import pdb; pdb.set_trace()
        
        uris_and_metadata = (
            p
            | "Create from df" >> beam.Create(df.to_dict("records"))
        )
        (
            uris_and_metadata
            | f"Ask {args.model}"
            >> beam.MapTuple(lambda uri, metadata: prompt(args.model, metadata, prompt_helper))
            | "Convert to JSONL" >> beam.Map(lambda dict: json.dumps(dict))
            | "Write Batches"
            >> beam.ParDo(
                StreamIntoFiles(
                    args.output_path,
                    file_name_suffix=".jsonl",
                    max_records_per_file=50,
                )
            )
        )


if __name__ == "__main__":
    main()
