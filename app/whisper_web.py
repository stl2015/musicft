import gradio as gr
import whisper


def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]


def main():
    ## sources = "upload" or "microphone"
    audio_input = gr.components.Audio(sources="upload", type="filepath")
    output_text = gr.components.Textbox()
    
    iface = gr.Interface(fn=transcribe_audio, inputs=audio_input, 
                         outputs=output_text, title="Audio Transcription App",
                         description="Upload an audio file and hit the 'Submit'\
                             button")
    
    iface.launch()


if __name__ == '__main__':
    main()