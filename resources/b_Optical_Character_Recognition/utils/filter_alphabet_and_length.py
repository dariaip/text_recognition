import numpy as np


def filter_alphabet_and_length(text: str, alphabet: str, max_length: int) -> str:
    """
    Вспомогательная функция для фильтрации входного текста по алфавиту и максимальной длине. 
    Все символы из входного текста, не входящие в алфавит, будут удалены, а длина текст будет уменьшена 
    с учетом пробелов в тексте (чтобы не обрезать слова).
    Например:
    filter_alphabet_and_length("123 hello ./@#$ привет", "123hello ./@#$", 13) вернет строку "123 hello".

    Args:
        text: входной текст
        alphabet: алфавит, по которому текст будет фильтроваться
        max_length: максимальная длина результирующего текста

    Returns:
        отфильтрованный и обрезанный по длине текст
    """
    raw_split = text.split()
    if len(raw_split) > 1:
        start_word_ind = np.random.randint(0, len(raw_split)-1)
        end_word_ind = np.random.randint(start_word_ind, len(raw_split))
        text = ' '.join(raw_split[start_word_ind:end_word_ind])
    curr_length = np.random.randint(2, max_length)
    filtered = ''.join(c for c in text if c in alphabet).strip()
    if len(filtered) < curr_length:
        return filtered
    split = filtered.split()
    new_text = ''
    curr_i = 0
    while curr_i < len(split) and len(new_text) < curr_length - len(split[curr_i]):
        new_text += split[curr_i] + ' '
        curr_i += 1
    return new_text.strip()
