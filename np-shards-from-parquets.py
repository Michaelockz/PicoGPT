import pyarrow.parquet as pq
import tiktoken
import numpy as np

# local parquets on my computer
parquets = [
    "Data/Fineweb-edu-10B/000_00000.parquet",
    "Data/Fineweb-edu-10B/001_00000.parquet",
    "Data/Fineweb-edu-10B/002_00000.parquet",
    "Data/Fineweb-edu-10B/003_00000.parquet",
]
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize_unit16(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


batch_size = 65536
for parq_index, parq in enumerate(parquets):
    pq_file = pq.ParquetFile(parq)
    for batch_index, batch in enumerate(pq_file.iter_batches(batch_size,columns=["text"])):
        df_chunk = batch.to_pandas()
        text_chunk = "".join(df_chunk["text"])
        tokens_np = tokenize_unit16(text_chunk)
        if batch_index >= 0 and batch_index < 4 and parq_index == 0:
            write_datafile(f"Data/FineWeb-Edu-NP/edufineweb-val-{parq_index:03d}-{batch_index:04d}", tokens_np)
        else:
            write_datafile(f"Data/FineWeb-Edu-NP/edufineweb-train-{parq_index:03d}-{batch_index:04d}", tokens_np)
        print("batch written.")
        del df_chunk, text_chunk
