from typing import List
import torch # type: ignore
from huggingface_hub import login, try_to_load_from_cache, snapshot_download # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
from typing import List
from pydantic import BaseModel  # type: ignore
from huggingface_hub import whoami, list_repo_files # type: ignore

_user_info = whoami()
print(f"Logged in as: {_user_info['name']}")

print("Loading model...")
model_id = "meta-llama/Llama-3.1-8B-Instruct" # Use this AFTER you have access
_tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
_model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype=torch.float16,
  device_map="auto",
  low_cpu_mem_usage=True,
  token=True
)
print("Model loaded!")


# Define a Pydantic model for your input if desired
class TextGenerationMessage(BaseModel):
  role: str  # e.g., "user", "system"
  content: str

def text_generation(messages: List[TextGenerationMessage]):
  try:
    # 1. Apply the chat template
    # The model's tokenizer knows how to format messages correctly (e.g., with [INST] tags)
    formatted_prompt = _tokenizer.apply_chat_template(
      messages,
      tokenize=False,  # We don't want to tokenize yet, just get the formatted string
      add_generation_prompt=True
    )

    # 2. Tokenize the formatted prompt
    inputs = _tokenizer(formatted_prompt, return_tensors="pt").to(_model.device)

    # 3. Generate
    with torch.no_grad():
      outputs = _model.generate(
        **inputs,
        max_new_tokens=150,  # Generate more tokens for a better answer
        do_sample=True,      # Enable sampling for less repetitive text
        temperature=0.7,     # Control randomness
        top_p=0.9,           # Nucleus sampling
        # eos_token_id=tokenizer.eos_token_id # Optional: define a stop token
      )

    # 4. Decode only the new tokens, skipping the input prompt
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response_text.strip()

  except Exception as e:
    return f"Error during text generation: {e}"
