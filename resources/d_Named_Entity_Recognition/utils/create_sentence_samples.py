from typing import List, Tuple


def shift_entities(
    sents_bounds: List[Tuple[int, int, str]], 
    entities: List[Tuple[int, int, str]]
):
  """
  Функция определяет сдвиг координат сущностей на уровень предложения.
  """
  ent_per_sents = []
  for sents_id, (start, end, sent) in enumerate(sents_bounds):
    ent_per_sent = []
    for start_e, end_e, ent_t in entities:

      if start <= start_e < end and start <= end_e <= end + 1:
        ent_per_sent.append((start_e - start, end_e - start, ent_t))
      # проверка на некорректность сегментации и попадание сущности на границу предложений
      if start <= start_e < end and end_e > end + 2:
        print(f'Entity {start_e, end_e, ent_t} was skipped due to '
              f'incorrect segmentation in sentence #{sents_id, sent}')

    ent_per_sents.append(ent_per_sent)
  return ent_per_sents
