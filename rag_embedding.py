import os
import fitz
from tqdm import tqdm
import random
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer



pdf_path = "intro_to_algorithms.pdf"

def text_formatter(text: str)-> str:
    """Performs minor formatting on the text."""
    cleaned_text = text.replace('\n', " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path: str)->list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({'page_number':page_number-22, 
                                'page_char_count':len(text),
                                "page_word_count":len(text.split(' ')),
                                "page_sentence_count_raw":len(text.split('. ')),
                                "page_token_count":len(text)/4,
                                "text":text})
    
    return pages_and_texts

pages_and_texts =  open_and_read_pdf(pdf_path)
# print(pages_and_texts[34:36])




# Further Text Processing (Splitting Pages into sentences)

# Two ways to do it
# 1. We've done this    by splitting on `". "`.
# 2. We can do this with a NLP library such as spacy and nltk.


nlp = English()
#Add a sentencizer pipeline, https://spacy.io/api/sentencizer
nlp.add_pipe('sentencizer')
# Example
# doc = nlp("This is a sentence. This another sentence. I like elephants.")
# assert len(list(doc.sents)) == 3
# print(list(doc.sents)) --> [This is a sentence., This another sentence., I like elephants.]
for item in tqdm(pages_and_texts):
    item['sentences'] = list(nlp(item['text']).sents)
    # Make sure all the sentences are strings (the default type is a spacy datatype)
    item['sentences'] = [str(sentence) for sentence in item['sentences']]
    # count the sentences
    item['page_sentence_spacy_count'] = len(item['sentences'])


# Chunking our sentences together
# The concept of splitting larger pieces of text into smaller ones is often reffered to as text splitting or chunking
# We'll keep it simple into groups of 10 sentences (however, you could also try 5,7,8,9 whatever you like)
# There are frameworks such as Langchain which help with this, however we'll stick with Python for now.
# Why we do this 
# So our texts are easier to filter (smaller groups of text can be easier to inspect than larger passage of text)
# so our text chunks can fit into our embedding model context window (e.g 384 tokens as limit)
# so our contexts passed to an LLM can be more specific and focused.
# 
# 
# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10

# create a function to split lists of texts recursively into chunk size
# e.g [20] - > [10,10] or [25] -> [10,10,5]
def split_list(input_list: list[str], slice_size: int) -> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

# test_list = list(range(25))
# print(split_list(test_list))

for item in tqdm(pages_and_texts):
    item['sentence_chunks'] = split_list(input_list=item['sentences'], slice_size=num_sentence_chunk_size)
    item['num_chunks'] = len(item['sentence_chunks'])

print(random.sample(pages_and_texts, k=1))
df  =pd.DataFrame(pages_and_texts)
print(df.head())   
print(df.describe().round(2)) 

# Splitting each chunk into it's own item
# we'd like to embed each chunk of sentences into its own numerical representation.
# That'll give us a good level of granularity.
# Meaning, we can dive specifically into text sample

pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chuncks in item['sentence_chunks']:
        chunck_dict = {}
        chunck_dict['page_number'] = item['page_number']
        if item['page_number'] == 739:
            print(item,'\n\n')
        # joi the sentences together into a paragraph like structure, aka join the list of sentences into one paragraph
        joined_sentence_chunk = "".join(sentence_chuncks).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])',r'. \1',joined_sentence_chunk) # ".A" -> ". A" (will work for any capital letter)

        chunck_dict['sentence_chunk'] = joined_sentence_chunk
        # get some stats on our chunk
        chunck_dict['chunk_char_count'] = len(joined_sentence_chunk)
        chunck_dict['chunk_word_count'] = len([word for word in joined_sentence_chunk.split(' ')])
        chunck_dict['chunk_token_count'] = len(joined_sentence_chunk)/4
        pages_and_chunks.append(chunck_dict)
    
print(len(pages_and_chunks))
print(random.sample(pages_and_chunks, k=1))
df  =pd.DataFrame(pages_and_chunks)
print(df.head())   
print(df.describe().round(2)) 

# Filter chunks of text for short chunks
# These chunks may not cpntain much useful information
# Show random chunks with under 30 tokens in length
min_token_length = 30
for row in df[df['chunk_token_count'] <= min_token_length].sample(5).iterrows():
    print(f"Chunk token count: {row[1]['chunk_token_count']} | Text: {row[1]['sentence_chunk']}")

# Filter our dataframe for rows with under 30 tokens
pages_and_chunks_over_min_token_len = df[df['chunk_token_count'] > min_token_length].to_dict(orient='records')
print(pages_and_chunks_over_min_token_len[:2])
print(random.sample(pages_and_chunks_over_min_token_len, k=1))
df  =pd.DataFrame(pages_and_chunks_over_min_token_len)
print(df.head())   
print(df.describe().round(2)) 
df = df.dropna()

embeddings_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device='cpu')

# # Create of list of sentences
# example_sentences = ["This is the embedding part, which will embed the given sentence",
#                      "Sentences can be embedded one by one or by the list",
#                      "I like Astronomy"]

# embeddings = embeddings_model.encode(example_sentences)
# embedding_dict = dict(zip(example_sentences, embeddings))

for item in tqdm(pages_and_chunks_over_min_token_len):
    item['embedding'] = embeddings_model.encode(item['sentence_chunk'],batch_size=8, convert_to_tensor=True)

text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

# If your embedding database is really large (e.g over 100k-1M samples) you might want to look into using a vector database for storage.