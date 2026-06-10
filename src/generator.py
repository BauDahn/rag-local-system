# src/generator.py
import os
import yaml
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.set_verbosity_error()

def run_rag_pipeline(query: str, config_path: str = "config.yaml") -> str:
    """
    Retrieves relevant text context from the local database and generates a 
    strictly bounded answer using a local quantized Hugging Face LLM.
    """
    # 1. Load the central system configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Re-initialize our working local vector store
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=config["models"]["embedding"],
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    db = Chroma(
        persist_directory=config["paths"]["vector_store"], 
        embedding_function=embeddings
    )

    # 3. Search the local database for matching text chunks
    print(f"[Generator] Searching database for context relevant to: '{query}'")
    matching_docs = db.similarity_search(query, k=3)
    context = "\n---\n".join([doc.page_content for doc in matching_docs])

    # 4. Configure advanced 4-bit quantization weights
    print(f"[Generator] Loading quantized LLM: {config['models']['llm']}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4", # Normalized Float 4: optimized for neural network weights
        bnb_4bit_use_double_quant=True
    )

    # 5. Load the tokenizer and quantized model from local cache/Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["llm"])
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["llm"],
        quantization_config=quantization_config,
        device_map="auto" # Dynamically maps model layers across available CPU/GPU memory
    )

    # 6. Design the strict context-bounded instruction prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict technical assistant. Answer the user's question USING ONLY "
                "the provided context sections. If the answer cannot be found in the context, "
                "say 'I do not know'. Do not make up or extrapolate any information."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        }
    ]

    # Format the prompt using the model's exact architectural chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 7. Execute local inference using a pipeline wrapper
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    outputs = generator(
        prompt,
        max_new_tokens=256,
        temperature=config["parameters"]["temperature"],
        do_sample=True
    )

    # Clean and extract only the new assistant response text
    generated_text = outputs[0]["generated_text"]
    response = generated_text[len(prompt):].strip()

    return response

if __name__ == "__main__":
    # Test the entire retrieval-generation loop end-to-end
    print("Testing Complete Quantized RAG Pipeline...")
    
    sample_query = "What does the narrator say about the character's costume?"
    
    try:
        answer = run_rag_pipeline(sample_query)
        print("\n================ SYSTEM ANSWER ================")
        print(answer)
        print("===============================================")
    except Exception as e:
        print(f"\n[Execution Failure]: {str(e)}")