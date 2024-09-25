# Copyright 2023 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict

import torch
import transformers

from m2t.arguments import TrainingArguments


def get_autocast_type(training_args):
    if training_args.bits == 16:
        if training_args.bf16:
            return torch.bfloat16
        elif training_args.fp16:
            return torch.float16
    else:
        logging.warning(
            f"Autocast logic for bits {training_args.bits} not implemented;"
            "falling back to torch.float. If there are type mismatch errors"
            "this could be the cause."
        )
        return torch.float


def get_compute_dtype(training_args: TrainingArguments):
    return (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if hasattr(trainer, 'deepspeed'):
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def gen_pkl_single(
        emb_file="tmp/clap-test/1788-start0.000-end25.000.npy",
        instruct_file="datasets/musicnet/4ca764dc5519488caf410af5c6c2df6f.jsonl",
        pkl_output="temp_train1.pkl",
        tar_output="temp_train1_archive.tar",
    ):
    import numpy as np

    data = np.load(emb_file)
    # shink dim temporarily: no need after the fix. remove after confirmed.
    # data = data.reshape(-1,)
    assert len(data.shape) == 2, 'check emb_dim'

    # load instruct data
    prompt_text="Describe the provided audio in details."

    import json
    with open(instruct_file, "rb") as f:
        response = json.load(f)

    id = response['uri']

    json_data={}
    json_data["response"] = [{"question": prompt_text, "answer": response['response']['caption']}]

    elem={}

    # combined together

    elem["audio_encoding.pyd"]=data
    elem["__key__"]=id
    elem["json"]={"response": json_data['response']}

    import pickle
    with open(pkl_output, 'wb') as file:
        pickle.dump(elem, file)
    
    import tarfile
    arcname = emb_file.split('/')[-1].split('.')[0]+'.pkl'
    print(arcname)
    with tarfile.open(tar_output, "w") as tar:
        tar.add(pkl_output, arcname=arcname)


import os

def find_ext_files(directory, ext='.mid'):
    # This list will store all the paths of .mid files.
    mid_files = []

    # os.walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up.
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                # os.path.join() constructs a pathname out of one or more partial pathnames.
                full_path = os.path.join(root, file)
                mid_files.append(full_path)

    return mid_files


import shutil

def move_files(ext = 'wav', source_directory = '/path/to/source',
               destination_directory = '/path/to/destination'):

# Get a list of all files in the source directory
    files = find_ext_files(source_directory, ext)
    print('[INFO] find {} files'.format(len(files)))

    # Move each file from the source directory to the destination directory
    for file in files:
        file_name = file.split('/')[-1]
        destination_path = os.path.join(destination_directory, file_name)
        shutil.copy(file, destination_path)


import pandas as pd
def find_unprocessed_wav_files(processed_jsonl, in_dir, out_dir):

    files = find_ext_files(in_dir, '.wav')
    print('[INFO] total files ', len(files))

    processed = pd.read_json(path_or_buf=processed_jsonl, lines=True)

    processed_ids = list(processed.id)

    processed_files = [id+'.wav' for id in processed_ids]

    for file_path in files:
        file = file_path.split('/')[-1]
        if file not in processed_files:
            shutil.copy(file_path, out_dir)