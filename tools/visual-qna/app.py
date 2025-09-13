from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import gradio as gr

model_id = "dandelin/vilt-b32-finetuned-vqa"

# load pretrained model and image processor
processor = ViltProcessor.from_pretrained(model_id)
model = ViltForQuestionAnswering.from_pretrained(model_id)

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Visual Question Answering with ViLT
    This demo uses the [ViLT model](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) to perform visual question answering.
    - Paste an image URL and a question about the image, then click "Submit" to get the answer.
    - You can type "exit" or "quit" in the question box to stop the interaction.
    """
    )

    with gr.Row():
        with gr.Column():
            image_data = gr.Image(type="numpy", label="Upload an image")
            question = gr.Textbox(label="Question", placeholder="Type your question here...")
            submit_btn = gr.Button("Submit")
        with gr.Column():
            output = gr.Textbox(label="Answer")

    def answer_question(image_data, question):
        if len(question) == 0:
            question = "what is the color of the cat (the right one)?"

        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            image = Image.fromarray(image_data).convert("RGB")

        encoding = processor(image, question, return_tensors="pt")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return model.config.id2label[idx]

    submit_btn.click(fn=answer_question, inputs=[image_data, question], outputs=output)

if __name__ == "__main__":
    demo.launch(
        pwa=True,
        share=False,        # Don't create a public link
        debug=False,        # Debug mode off for a production launch
        server_name="0.0.0.0", # Allow access on local network
        server_port=9000,   # Run on a custom port
        # auth=("admin", "123") # Uncomment for password protection
    )