import os

from transformers import AutoModelForCausalLM, AutoTokenizer


#HUGGING_FACE_API_KEY = os.environ.get("HUGGINGFACE_KEY")

acces_token = "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"
# The device to load the model onto:
#
#Available device types:
#"cuda" - NVIDIA GPU
#"cpu" - Plain CPU
# "mps" -. Apple Silicon

device = "cuda"

model =AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-2-12b-chat", token= acces_token)
model.to(device)


tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-2-12b-chat", token=acces_token)

