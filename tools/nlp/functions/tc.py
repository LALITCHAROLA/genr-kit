from typing import List
import torch # type: ignore
from huggingface_hub import login, try_to_load_from_cache, snapshot_download # type: ignore
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline # type: ignore
from typing import List
from pydantic import BaseModel  # type: ignore
from huggingface_hub import whoami, list_repo_files # type: ignore

_user_info = whoami()
print(f"Logged in as: {_user_info['name']}")

print("Loading model...")
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
_tokenizer = DistilBertTokenizer.from_pretrained(model_id)
_model = DistilBertForSequenceClassification.from_pretrained(model_id)
print("Model loaded!")


""" Text classification """
def text_classification(text: str):
  try:
    inputs = _tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return _model.config.id2label[predicted_class_id]
  except Exception as e:
    return f"Error: {e}"
