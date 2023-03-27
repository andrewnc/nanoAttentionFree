import os
import tiktoken
import numpy as np
import jsonlines
import requests
import gc

def load_data(input_file_path):
    print("loading data")
    with jsonlines.open(input_file_path) as reader:
        data = []
        for obj in reader:
            data.append(obj['table'])
    return ''.join(data)

def encode_data_in_chunks(data, chunk_size=10000):
    print("encoding data")
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")

    train_token_count = 0
    with open(train_file_path, 'ab') as f:
        for i in range(0, len(train_data), chunk_size):
            train_ids_chunk = enc.encode_ordinary(train_data[i:i+chunk_size])
            train_token_count += len(train_ids_chunk)
            np.array(train_ids_chunk, dtype=np.uint16).tofile(f)
    
    val_token_count = 0
    with open(val_file_path, 'ab') as f:
        for i in range(0, len(val_data), chunk_size):
            val_ids_chunk = enc.encode_ordinary(val_data[i:i+chunk_size])
            val_token_count += len(val_ids_chunk)
            np.array(val_ids_chunk, dtype=np.uint16).tofile(f)

    print(f"train has {train_token_count:,} tokens")
    print(f"val has {val_token_count:,} tokens")

    # return train_ids, val_ids

def save_to_bin_files(train_ids, val_ids, train_file_path, val_file_path):
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(train_file_path)
    val_ids.tofile(val_file_path)

if __name__ == "__main__":
    input_file_path = os.path.join(os.path.dirname(__file__), 'csvs.jsonl')
    train_file_path = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_file_path = os.path.join(os.path.dirname(__file__), 'val.bin')

    data = load_data(input_file_path)
    # train_ids, val_ids = encode_data_in_chunks(data)
    # save_to_bin_files(train_ids, val_ids, train_file_path, val_file_path)# ping ntfy.sh
    requests.post("http://ntfy.sh/4CrhOZDCb1ojuzCa", data=f"gittables/prepare.py finished: token count {len(train_ids):,}")
# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
