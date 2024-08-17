import logging
import sys
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# setup prompts - specific to StableLM
from llama_index.core import PromptTemplate

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM assists users by retrieving and summarizing relevant information efficiently.
- StableLM is designed to provide concise and accurate responses to user queries.
- StableLM can handle a variety of queries and will summarize information to deliver clear and useful responses.
- StableLM will refuse to engage in any harmful activities or generate inappropriate content.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
# Initialize HuggingFace LLM with specific settings
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": True},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="stabilityai/stablelm-2-12b-chat",
    model_name="stabilityai/stablelm-2-12b-chat",
    device_map="cuda",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.float16}
)

embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5", embed_batch_size=100
    )

# Set global settings
Settings.llm = llm
Settings.embed_model = embed_model


postprocessor = SentenceEmbeddingOptimizer(
    embed_model=embed_model,
    percentile_cutoff=0.5,
    threshold_cutoff =0.59,
)
# Load the existing index
storage_context_vector = StorageContext.from_defaults(persist_dir="./Total_Data_embed")
index = load_index_from_storage(storage_context=storage_context_vector)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=8,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()



# Main function
if __name__ == "__main__":
    logging.info("Loading existing embeddings and preparing for querying...\n")

    primary_chat_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[postprocessor],
    response_mode = 'compact',
)

    while True:
        query = input("Enter your query (or type 'quit' to stop): ").strip()
        if query.lower() == "quit":
            print("Exiting the query engine.")
            break

        try:
            response = primary_chat_engine.query(query)
        except ValueError as e:
            if "Optimizer returned zero sentences" in str(e):
                logging.info(
                    "Optimizer found zero relevant sentences. Initiating fallback chat engine..."
                )
                response = llm.complete(query)
            else:
                raise  # Re-raise the exception if it's not the one we're looking for
        except Exception as e:
            logging.error(f"Unexpected error occurred: {str(e)}")
            response = "An error occurred, please try again later."

        print("Response:", response)