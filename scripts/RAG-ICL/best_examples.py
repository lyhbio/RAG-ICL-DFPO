import json
import argparse
import os
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def ensure_directory(directory):
    """Ensure the existence of a directory."""
    os.makedirs(directory, exist_ok=True)
    print(f"Directory {directory} is ready.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Utilizing RAG to Isolate High-Quality Examples")
    
    parser.add_argument("--task", 
                        type=str, 
                        required=True, 
                        choices=['ner', 're'], 
                        help="Type of dataset"
                        )
    
    parser.add_argument("--one_dataset", 
                        type=str, 
                        required=True, 
                        choices=['nlm_gene', 'ncbi_disease', 'ddi_corpus', 'chemdner', 'biorelex', 'bc5cdr'], 
                        help="The name of the dataset to use."
                        )
    
    parser.add_argument("--embedding_model_path", 
                        type=str, 
                        required=True,
                        help="Path of the used embedding model"
                        )
    
    parser.add_argument("--search_type", 
                        type=str, 
                        default="similarity", 
                        choices=['similarity', 'mmr', 'similarity_score_threshold']
                        )
    
    parser.add_argument("--k", 
                        type=int, 
                        default=30,
                        help="Sample size for one-stage retrieval"
                        )
    
    parser.add_argument("--score_threshold", 
                        type=float, 
                        default=0.79
                        )
    
    parser.add_argument("--rerank_model_path", 
                        type=str, 
                        required=True,
                        help="Path of the used rerank model"
                        )
    
    return parser.parse_args()

def load_dataset(data_path, task, dataset_name):
    """Load training and test datasets."""
    dataset_base_path = f"{data_path}/{task}/{dataset_name}/{dataset_name}_{{}}_processed.json"
    if dataset_name == "bc5cdr":
        dataset_base_path = f"{data_path}/ner/bc5cdr_chemical/bc5cdr_chemical_{{}}_processed.json"
    with open(dataset_base_path.format("train"), 'r', encoding='utf-8') as f:
        trainset = json.load(f)
    with open(dataset_base_path.format("test"), 'r', encoding='utf-8') as f:
        testset = json.load(f)
    return trainset, testset

def load_models(embedding_model_path, rerank_model_path):
    """Load embedding and rerank models."""
    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'batch_size': 4, 'normalize_embeddings': True})
    tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        rerank_model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="auto")
    return embed_model, tokenizer, model

def main():
    args = parse_args()
    data_path = "../../data/collate"
    trainset, testset = load_dataset(data_path, args.task, args.one_dataset)

    # Prepare passages and metadata
    passages = [value[0] for value in trainset.values()]
    meta_ID = [{"ID": key} for key in trainset.keys()]

    # Load models
    embed_model, tokenizer, rerank_model = load_models(args.embedding_model_path, args.rerank_model_path)

    # Create and save vectorstore
    vectorstore_path = "../../RAG-ICL/vectorstore"
    result_path = "../../RAG-ICL/result"
    ensure_directory(vectorstore_path)
    ensure_directory(result_path)

    faiss_vectorstore = FAISS.from_texts(passages, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, metadatas=meta_ID)
    faiss_vectorstore.save_local(f"{vectorstore_path}/{args.one_dataset}")
    print("vectorstore successfully saved locally")
    # Recall and rerank
    retriever = faiss_vectorstore.as_retriever(search_type=args.search_type, search_kwargs={"k": args.k}) # "score_threshold": args.score_threshold, 
    best_example = {}
    print("Start retrieval and rerank")
    for key, value in tqdm(testset.items()):
        query = value[0]
        related_passages = retriever.get_relevant_documents(query)
        sentence_pairs = [[query, passage.page_content] for passage in related_passages]

        # Tokenize inputs and send to device
        inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs_on_device = {k: v.to("cuda") for k, v in inputs.items()}

        # Calculate scores
        with torch.no_grad():
            scores = rerank_model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
            scores = torch.sigmoid(scores)
            indices = scores.topk(10).indices.cpu().detach().tolist()
        result_passages = [related_passages[index].metadata['ID'] for index in indices]
        best_example[key] = result_passages

    # Save results
    with open(f"{result_path}/{args.one_dataset}_best_example.json", 'w') as f:
        json.dump(best_example, f, indent=4)

if __name__ == "__main__":
    main()
