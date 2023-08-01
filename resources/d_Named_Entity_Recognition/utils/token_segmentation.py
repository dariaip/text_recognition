from nltk.tokenize import wordpunct_tokenize
from typing import List, Tuple

def token_segmentation(sentences_per_docs: List[List[str]]) -> List[List[List[str]]]:
  tokens_per_docs = []
  for text in sentences_per_docs:
    tokens_per_doc = []
    for sentence in text:
      tokens = wordpunct_tokenize(sentence)
      if tokens:
        tokens_per_doc.append(tokens)
    tokens_per_docs.append(tokens_per_doc)
  return tokens_per_docs
