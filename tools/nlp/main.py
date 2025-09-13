import sys
import os

# Add the 'functions' directory to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))

# from tg import text_generation, TextGenerationMessage
# from tc import text_classification
from ner import named_entity_recognition

""" Named Entity Recognition """
while True:
  text = input("Enter text for NER (or 'exit' to quit): ")
  if text.lower() in ["exit", "quit"]:
      break
  if len(text) == 0:
      text = "My name is Wolfgang and I live in Berlin"
  
  ner_results = named_entity_recognition(text)
  # print("Entities found:", ner_results)
  for e in ner_results:
    # print(e)
    print(f"{e['word']}: {e['entity']} ({e['score']:.4f})")


""" Text classification """
# while True:
#   text = input("Enter text for classification (or 'exit' to quit): ")
#   if text.lower() in ["exit", "quit"]:
#       break
#   if len(text) == 0:
#       text = "I love using transformers library!"
  
#   tc_result = text_classification(text)
#   print(tc_result)


""" Text generation """
# while True:
#   user_input = input("Ask anything (or 'exit' to quit): ")
#   if user_input.lower() in ["exit", "quit"]:
#       break
#   if len(user_input) == 0:
#       user_input = "Hello, how are you?"
  
#   messages = [
#     TextGenerationMessage(role="system", content="You are a helpful assistant."),
#     TextGenerationMessage(role="user", content=user_input),
#   ]
#   tg_result = text_generation(messages)
#   print("AI:", tg_result)