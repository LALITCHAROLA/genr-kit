from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

"""
*Instructions: 
This script performs visual question answering using a pre-trained ViLT model.
It prompts the user to input an image URL and a question about the image, then outputs the model's answer.
To run the script, ensure you have the required libraries installed: transformers, requests, and Pillow.

*Use cases:
1. Visual Question Answering: Users can ask questions about the content of an image, and the model will provide answers based on its understanding of the image.
2. Interactive Image Analysis: This tool can be used in applications where users need to interactively analyze images by asking questions.
3. Educational Purposes: It can serve as a demonstration of how multi-modal models work, combining visual and textual data to generate responses.
"""

model_id = "dandelin/vilt-b32-finetuned-vqa"

# load pretrained model and image processor
processor = ViltProcessor.from_pretrained(model_id)
model = ViltForQuestionAnswering.from_pretrained(model_id)

# prepare image + question
url = input("Paste the image URL: ")
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

while True:
    text = input("> ")
    if text.lower() in ["exit", "quit"]:
        break
    if len(text) == 0:
        text = "what is the color of the cat (the right one)?"

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print(model.config.id2label[idx])

