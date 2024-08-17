import logging
import sys
import torch
from transformers import AutoTokenizer
from llama_index.core.settings import Settings
from llama_index.llms.huggingface import HuggingFaceLLM

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Setup system prompt specific to StableLM
system_prompt = """# StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM assists users by retrieving and summarizing relevant information efficiently.
- StableLM is designed to provide concise and accurate responses to user queries.
- StableLM can handle a variety of queries and will summarize information to deliver clear and useful responses.
- StableLM will refuse to engage in any harmful activities or generate inappropriate content.
"""

# Initialize HuggingFace LLM with specific settings
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": True},
    system_prompt=system_prompt,
    tokenizer_name="stabilityai/stablelm-2-12b-chat",
    model_name="stabilityai/stablelm-2-12b-chat",
    device_map="cuda",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.float16}
)

# Set global settings
Settings.llm = llm

# Main function
if __name__ == "__main__":
    logging.info("LLM initialized and ready for querying...\n")

    while True:
        query = input("Enter your query (or type 'quit' to stop): ").strip()
        if query.lower() == "quit":
            print("Exiting the query engine.")
            break

        try:
            response = llm.complete(query)
        except Exception as e:
            logging.error(f"Unexpected error occurred: {str(e)}")
            response = "An error occurred, please try again later."

        print("Response:", response)
