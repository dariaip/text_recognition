from pathlib import Path
from typing import List

from tqdm.auto import tqdm

from .filter_alphabet_and_length import filter_alphabet_and_length


def make_sentences_from_wiki(wiki_folder: str, alphabet: str, max_length: int) -> List[str]:
    """
    Вспомогательная функция для извлечения и фильтрации строк с текстом из файлов со статьями Википедии.

    Args:
        wiki_folder: путь к файлам со статьями Википедии
        alphabet: алфавит, по которому текст будет фильтроваться
        max_length: максимальная длина результирующего текста

    Returns:
        список отфильтрованных строк из статей Википедии
    """
    all_sentences = []
    txt_wiki = list(map(str, Path(wiki_folder).glob('*.txt')))
    for txt_file in tqdm(txt_wiki):
        with open(txt_file, 'r') as r:
            data = [line.strip() for line in r.readlines()]
        for line in data:
            sentences = line.split('.')
            sentences = [filter_alphabet_and_length(s + '.', alphabet=alphabet, max_length=max_length) 
                         for s in sentences]
            sentences = [s for s in sentences if len(s) > 0 and s != '.']
            all_sentences.extend(sentences)
    return all_sentences
