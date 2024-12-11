import pandas as pd
import random
import torch
import numpy as np
from pprint import pprint

# Rag goal : Retrieve relevant passages based on query and use those passages to augument an input to a LLM so it can generate an output based on those relevant passages.

# Similarity Search
# Embedding can be used almost any type of data: images -> emb, text -> emb, sound -> emb
# Comparing embeddings is known as similarity search, vector search, semantic search
# In our case, we want to query our Introduction to Algorithms book passages based on the semantic or *vibe*
# So if i search for "Binary search" i should get relevant passage to that text.
# Whereas with keyword search, If i search "Apple" I get back passages with specifically "Apple".

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import text and embedding df
text_chunks_and_embeddings_df = pd.read_csv('text_chunks_and_embeddings_df.csv')
print(text_chunks_and_embeddings_df['embedding'])
nan_rows2 = text_chunks_and_embeddings_df[text_chunks_and_embeddings_df['embedding'].isna()].to_html('es.html')
print(nan_rows2);exit()
# convert embedding column back to np.array (it got converted to string when it saved to csv)
text_chunks_and_embeddings_df['embedding'] = text_chunks_and_embeddings_df['embedding'].apply(lambda x: np.fromstring(x))

# convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient='records')


embeddings = text_chunks_and_embeddings_df['embedding'].to_list()
