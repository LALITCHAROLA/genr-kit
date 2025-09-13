from typing import List
import torch # type: ignore
from huggingface_hub import login, try_to_load_from_cache, snapshot_download # type: ignore
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline # type: ignore
from typing import List
from pydantic import BaseModel  # type: ignore
from huggingface_hub import whoami, list_repo_files # type: ignore

_user_info = whoami()
print(f"Logged in as: {_user_info['name']}")

print("Loading model...")
model_id = "dslim/bert-base-NER"
_tokenizer = AutoTokenizer.from_pretrained(model_id)
_model = AutoModelForTokenClassification.from_pretrained(model_id)
print("Model loaded!")


""" Named Entity Recognition """
def named_entity_recognition(text: str):
  try:
    nlp = pipeline("ner", model=_model, tokenizer=_tokenizer)
    return nlp(text)
  except Exception as e:
    return f"Error: {e}"
