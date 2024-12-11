from spacy.lang.en import English

nlp = English()

#Add a sentencizer pipeline, https://spacy.io/api/sentencizer
nlp.add_pipe('sentencizer')

# Example
# doc = nlp("This is a sentence. This another sentence. I like elephants.")
# assert len(list(doc.sents)) == 3
# print(list(doc.sents)) --> [This is a sentence., This another sentence., I like elephants.]