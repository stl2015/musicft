import glob
import os
import gradio as gr

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from m2t.arguments import DataArguments, ModelArguments, TrainingArguments
from m2t.conversation_utils import extract_response_tokens
from m2t.data_modules import make_mm_config
from m2t.infer_from_prompt import infer_with_prompt
from m2t.models.utils import load_pretrained_model
from m2t.tokenizer import get_prompt_end_token_sequence
from m2t.utils import get_autocast_type
from scripts.clap.clap_embeddings import gen_audio_embeddings_app
from jukebox.dataflow_inference import gen_jukebox_embedding_app

data_args = DataArguments()
model_args = ModelArguments(model_name_or_path='checkpoint/meta-llama-jukebox/')
training_args = TrainingArguments(output_dir='tmp')

max_new_tokens = 1024

def load_model(data_args, model_args, ckpt_num):
    
    model, tokenizer = load_pretrained_model(model_args.model_name_or_path, ckpt_num=ckpt_num)
    model.cuda()

    data_args.mm_use_audio_start_end = True

    end_seq = get_prompt_end_token_sequence(tokenizer, model_args.model_name_or_path)

    return model, tokenizer, end_seq


model, tokenizer, end_seq = load_model(
    data_args=data_args, model_args=model_args, ckpt_num=0)


def transcribe_audio(
    audio_file: str,
):
    
    prompt = "Describe the contents of the provided audio in detail."
    max_samples = None
        
    output_dir = 'tmp/app-test'
    ## default to use jukebox
    gen_jukebox_embedding_app(input_file=audio_file, output_dir=output_dir)

    audio_encodings = glob.glob(os.path.join(output_dir, "*.npy"))

    multimodal_cfg = make_mm_config(data_args)

    with torch.autocast(device_type="cuda", dtype=get_autocast_type(training_args)):
        with torch.inference_mode():
            for i, encoding_fp in tqdm(enumerate(audio_encodings), total=max_samples):
                audio_encoding = np.load(encoding_fp)

                print(f"[DEBUG] inferring with fixed prompt: {prompt}")
                outputs_i = infer_with_prompt(
                    prompt,
                    model=model,
                    audio_encoding=audio_encoding,
                    multimodal_cfg=multimodal_cfg,
                    end_seq=end_seq,
                    tokenizer=tokenizer,
                    audio_first=True,
                    max_new_tokens=max_new_tokens,
                )

                print("[PROMPT]")
                print(prompt)

                print("[MODEL COMPLETION]")
                # input_and_model_completion_text = tokenizer.decode(outputs_i[0])
                model_completion_ids = extract_response_tokens(outputs_i[0], end_seq)
                model_completion_text = tokenizer.decode(model_completion_ids)
                print(model_completion_text)

                print("%" * 40)
                if max_samples and (i >= max_samples):
                    break

    return model_completion_text


def main():

    ## sources = "upload" or "microphone"
    audio_input = gr.components.Audio(sources="upload", type="filepath")
    output_text = gr.components.Textbox()
    
    iface = gr.Interface(fn=transcribe_audio, inputs=audio_input, 
                         outputs=output_text, title="Music Companion App",
                         description="Upload an audio file and hit the 'Submit'\
                             button")
    
    iface.launch(share=True)


if __name__ == '__main__':
    main()

