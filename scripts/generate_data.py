import argparse
import os
from datasets import load_dataset
import json

def download_dataset(dataset_name, save_path):
    print(f"Downloading {dataset_name} dataset...")
    dataset = load_dataset("bigbio/" + dataset_name, name=dataset_name + "_bigbio_kb", trust_remote_code=True)
    dataset_path = os.path.join(save_path, dataset_name)
    dataset.save_to_disk(dataset_path)
    print(f"Dataset saved to {dataset_path}.")
    return dataset

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f"Directory {directory} is ready.")

def setup_parser():
    """Create a command-line argument parser"""
    parser = argparse.ArgumentParser(description="Download specified biomedical datasets and save them to disk")
    parser.add_argument('--datasets', nargs='*', choices=[
        'nlm_gene', 'ncbi_disease', 'ddi_corpus', 'chemdner', 'biorelex', 'bc5cdr'],
        help='Select datasets to download')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    return parser.parse_args()

def process_ner(example, task_type=None):
    # Concatenate text from the first and last passage
    text = example['passages'][0]['text'][0] + " " + example['passages'][-1]['text'][0]
    # Filter entities if task_type is specified, otherwise use all entities
    entities = [e for e in example['entities'] if task_type is None or e['type'] == task_type]
    # Create a dictionary to handle case-insensitive uniqueness
    unique_items = {}
    for entity in entities:
        item = entity['text'][0]
        lower_item = item.lower()
        # Update the item if it's not in the dictionary or if the new item has preferable capitalization
        if lower_item not in unique_items or (item[0].isupper() and unique_items[lower_item][0].islower()):
            unique_items[lower_item] = item
    return {
        'document_id': example['document_id'],
        'text': text,
        'entity': list(unique_items.values())
    }

def process_re(example):
    combined_text = ' '.join(text for passage in example['passages'] for text in passage['text'])
    entities2id = {entitiy['id']:entitiy['text'][0] for entitiy in example['entities']}
    relation_data = [[entities2id.get(relation['arg1_id']), entities2id.get(relation['arg2_id'])] 
                    for relation in example['relations']]
    unique_relaton_list = []
    [unique_relaton_list.append(item) for item in relation_data if item not in unique_relaton_list]
    return {'document_id':example['document_id'],
            'text':combined_text,
            'entity':unique_relaton_list}

def save_dataset(dataset_type, dataset_dict, base_path):
    for key, data in dataset_dict.items():
        ensure_directory(f"{base_path}/{key}")
        result = {}

        for sample in data['train']:
            result[sample['document_id']] = [sample['text'], sample['entity']]
        with open(f"{base_path}/{key}/{key}_train_processed.json", 'w') as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)

        if dataset_type == 're' and key == "biorelex":
            test_data_key = 'validation'
        else:
            test_data_key = 'test'

        result = {}
        for sample in data[test_data_key]:
            result[sample['document_id']] = [sample['text'], sample['entity']]
        with open(f"{base_path}/{key}/{key}_test_processed.json", 'w') as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)

num_cores = os.cpu_count()

# Define the path for the raw data folder
data_path = "../data/raw"
ensure_directory(data_path)  # Ensure the raw directory exists

# Dictionary to store and access datasets
datasets = {}

# Mapping of dataset names
dataset_names = [
    'nlm_gene', 'ncbi_disease', 'ddi_corpus', 'chemdner', 'biorelex', 'bc5cdr'
]

args = setup_parser()
# Execute download based on parameters
if args.all:
    # Download all datasets
    for name in dataset_names:
        datasets[name] = download_dataset(name, data_path)
elif args.datasets:
    # Only download specified datasets
    for name in args.datasets:
        if name in dataset_names:
            datasets[name] = download_dataset(name, data_path)
else:
    print("No dataset specified. Please use --datasets to select datasets or --all to download all.")

# Initialize dictionary to store processed datasets
processed_ner_datasets = {}
ner_datasets_name = ['bc5cdr', 'nlm_gene', 'ncbi_disease', 'chemdner']
# Apply processing to all datasets except special cases
for name in ner_datasets_name:
    if name == 'bc5cdr':
        # Handle special cases for 'bc5cdr' dataset
        for task_type in ["Chemical", "Disease"]:
            processed_ner_datasets[f'bc5cdr_{task_type.lower()}'] = datasets['bc5cdr'].map(
                lambda x: process_ner(x, task_type=task_type),
                num_proc=num_cores//4,
                remove_columns=datasets['bc5cdr'].column_names['train']
            )
    else:
        # Apply general processing
        processed_ner_datasets[name] = datasets[name].map(
            process_ner,
            num_proc=num_cores//4,
            remove_columns=datasets[name].column_names['train']
        )

processed_re_datasets = {}
re_datasets_name = ['bc5cdr', 'ddi_corpus', 'biorelex']
# Apply processing to all datasets except special cases
for name in re_datasets_name:
    filtered_data = datasets[f"{name}"].filter(lambda x: x['entities'] != [], num_proc=num_cores//4)
    processed_re_datasets[name] = filtered_data.map(
    process_re,
    num_proc=num_cores//4,
    remove_columns=filtered_data.column_names['train']
    )  

save_dataset('ner', processed_ner_datasets, "../data/collate/ner")
save_dataset('re', processed_re_datasets, "../data/collate/re")