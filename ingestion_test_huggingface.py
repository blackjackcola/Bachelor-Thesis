import torch
import os

# for some additional information for troubleshooting
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)

from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig



llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    tokenizer_name="stabilityai/stablelm-2-12b-chat",
    model_name="stabilityai/stablelm-2-12b-chat",
    device_map="cuda",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.float16}
)

if __name__ == "__main__":
    logging.info("Going to process and store embeddings locally...\n")

    # You need to create an instance of the UnstructuredReader to use it

    logging.info("Loading documents from the directory...\n")

    dir_reader = SimpleDirectoryReader(
        input_dir="./totalData_inText"
    )
    documents = dir_reader.load_data()
    logging.info(f"Loaded {len(documents)} documents.")

    # Initialize the node parser and other settings
    logging.info("Setting up node parser and LLM settings...\n")

    parser = LangchainNodeParser(
        RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=50,
            separators=["<eos>"],  # Fügen Sie <eos> als Separator hinzu
            keep_separator=True,  # Behalten Sie den Separator am Ende jedes Abschnitts
            is_separator_regex=False,
        )
    )
    nodes = parser.get_nodes_from_documents(documents)
    pass

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5", embed_batch_size=100
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.nodes = nodes

    # Process documents and store embeddings
    logging.info("Processing documents and storing embeddings...\n")

    # this index will create a default storage from llamaindex VectorStore, if you want to use other Vector Store e.g Chromadb just uncomment storage context

    index = VectorStoreIndex.from_documents(
        documents=documents,
        settings=Settings,
        show_progress=True,
    )

    index.storage_context.persist("Total_Data_embed")
    logging.info("Finished processing and storing embeddings locally.\n")


