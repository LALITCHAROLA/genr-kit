# Genr Kit

The open-source playground for generative AI. Rapidly prototype and deploy multi-modal applications—from text and image classification to speech synthesis—using Python, Gradio, and Transformers.

## Features

- 🚀 **Gradio Interface**: Fast, modern AI-Powered apps interface with the API ready feature
- 🤗 **Hugging Face Integration**: Uses state-of-the-art BLIP model for image captioning
- 📦 **Batch Processing**: Support for processing multiple images at once
- 🔥 **GPU Support**: Automatic GPU acceleration when available

## Tools

Genr-Kit provides a comprehensive suite of tools for common generative AI tasks. Each tool leverages state-of-the-art, publicly available models from the Hugging Face Hub.

---

### 📝 Text & NLP Tools

#### **Text Classification**

Categorize text into predefined labels like sentiment or topic.

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
  A fast and accurate model fine-tuned for sentiment analysis (positive/negative), perfect for real-time applications.

#### **Text Generation**

Generate coherent and contextually relevant text from a prompt.
**Model:** `gpt2`
A pioneering transformer model capable of generating creative text continuations in various styles.

#### **Sentiment Analysis**

Analyze text to determine its emotional tone (e.g., positive, negative, neutral).
**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
A robust model trained on a large corpus of tweets, excellent for modern, informal language.

#### **Text Summarization**

Condense long articles, reports, or documents into concise summaries.
**Model:** `facebook/bart-large-cnn`
The BART model fine-tuned on the CNN/DailyMail dataset, making it excellent for abstractive summarization.

#### **Named Entity Recognition (NER)**

Identify and extract entities like names, organizations, and locations from text.
**Model:** `dslim/bert-base-NER`
A BERT model specifically fine-tuned to recognize common named entities with high accuracy.

#### **Text-to-Command**

Convert natural language instructions into structured commands or API calls.
**Model:** `microsoft/DialoGPT-medium`
While often used for chat, its fine-tuning capabilities make it a good base for learning instruction-to-command tasks.

---

### 🖼️ Computer Vision Tools

#### **Image Classification**

Identify the main subject or scene within an image.
**Model:** `google/vit-base-patch16-224`
A Vision Transformer (ViT) model that achieves excellent accuracy on the ImageNet-1k benchmark.

#### **Object Detection**

Locate and identify multiple objects within an image using bounding boxes.
**Model:** `hustvl/yolos-tiny`
A You Only Look Once (YOLO) transformer model that provides a great balance of speed and accuracy for real-time detection.

#### **Image Generation**

Generate novel images from a text description.
**Model:** `runwayml/stable-diffusion-v1-5`
A powerful latent diffusion model for creating high-quality, detailed images from any text prompt.

#### **Image Captioning**

Generate a descriptive English-language caption for a given image.
**Model:** `Salesforce/blip-image-captioning-base`
The BLIP model provides high-quality, context-aware captions, ideal for accessibility and content description.

#### **Image-to-Image Translation**

Transform an input image based on a text prompt (e.g., style transfer, enhancement).
**Model:** `timbrooks/instruct-pix2pix`
A model specifically fine-tuned for following instructions to edit images, like "make it a cartoon".

#### **Image Segmentation**

Identify and map specific objects or regions in an image at the pixel level.
**Model:** `facebook/detr-resnet-50-panoptic`
A transformer-based model that performs both instance and panoptic segmentation in a single architecture.

---

### 🔊 Audio & Speech Tools

#### **Text-to-Speech**

Convert written text into natural-sounding spoken audio.
**Model:** `espnet/kan-bayashi_ljspeech_vits`
A VITS-based model that produces highly natural and expressive speech in English.

#### **Speech-to-Text**

Transcribe spoken audio from various languages into written text.
**Model:** `openai/whisper-base`
OpenAI's Whisper model offers robust, accurate transcription and translation across multiple languages.

#### **Speech Enhancement**

Remove background noise and improve the clarity of an audio recording.
**Model:** `microsoft/speechbrain-mtl-mimic-voicebank`
A model trained specifically for speech enhancement and denoising tasks.

#### **Music Generation**

Generate short musical audio clips from text descriptions.
**Model:** `facebook/musicgen-small`
A simple and controllable model for generating high-quality music from text prompts.

---

### 🔗 Multi-Modal Tools

#### **Visual Question Answering**

Answer natural language questions about the contents of an image.
**Model:** `dandelin/vilt-b32-finetuned-vqa`
A vision-and-language transformer (ViLT) model designed for answering questions about images.

#### **Document Question Answering**

Answer questions based on the content of a document (e.g., scanned PDFs, images with text).
**Model:** `impira/layoutlm-document-qa`
A model that understands the layout of documents (text + spatial information) to answer questions accurately.

#### **Embeddings Generator**

Create numerical vector representations (embeddings) of text, images, or audio for analysis and search.
**Model:** `sentence-transformers/all-MiniLM-L6-v2`
A versatile model that maps sentences and paragraphs to a dense vector space, perfect for semantic search and clustering.

## Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/alinrajpoot/genr-kit.git
cd genr-kit
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
python main.py
```

The API will be available at `http://localhost:9000`

### Usage

Once the server is running, visit http://localhost:9000 in browser.

## Requirements

- Python 3.8+
- Gradio
- Hugging Face Transformers
- PyTorch
- Pillow (PIL)

## License

Open source - feel free to contribute and improve!
