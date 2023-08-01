import pytorch_lightning as pl
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple
from rusenttokenize import ru_sent_tokenize
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from tqdm.notebook import tqdm
import torch
from torch import nn


class CustomDataset(Dataset):
    """
    Класс Dataset для подготовки данных и подачи модели BERT.
    Словари для датасета RuRED.
    """

    def __init__(
            self, samples: list, pad: str = "O", pad_index: int = 0,
            max_seq_len: int = 512, tokenizer_name_or_path="sberbank-ai/ruBert-base"
    ):
        self.pad = pad
        self.pad_index = pad_index
        self.max_seq_len = max_seq_len
        self.label_to_id = {
            "REGION": 1,
            "CURRENCY": 2,
            "NORP": 3,
            "MONEY": 4,
            "GROUP": 5,
            "EVENT": 6,
            "QUANTITY": 7,
            "STREET": 8,
            "CITY": 9,
            "NATIONALITY": 10,
            "AGE": 11,
            "FAMILY": 12,
            "CARDINAL": 13,
            "TIME": 14,
            "ORGANIZATION": 15,
            "ORDINAL": 16,
            "LAW": 17,
            "FAC": 18,
            "PERSON": 19,
            "COUNTRY": 20,
            "GPE": 21,
            "PERCENT": 22,
            "LOCATION": 23,
            "DATE": 24,
            "PRODUCT": 25,
            "PROFESSION": 26,
            "WORK_OF_ART": 27,
            "RELIGION": 28,
            "BOROUGH": 29,
            self.pad: self.pad_index
        }

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path, max_len=10000)

        self.wpt = WordPunctTokenizer()

        self.tokens_per_sample = []
        self.data = [
            self.prepare_sentence(sentence, markup, self.tokens_per_sample)
            for (_, _, sentence), markup in samples
            if sentence.strip()
        ]

    def prepare_sentence(self, sentence, ents, tokens_per_sample):

        tokens_positions = list(self.wpt.span_tokenize(sentence))
        tokens = [sentence[s: e] for s, e in tokens_positions]
        tokens_per_sample.append([(s, e, t) for (s, e), t in zip(tokens_positions, tokens)])
        labels = [self.pad] * len(tokens_positions)

        if ents:
            for st_ent, end_ent, type_ent in ents:
                for index, (start, end) in enumerate(tokens_positions):
                    if st_ent <= start < end_ent:
                        labels[index] = type_ent

        labels_ids = [-100]
        token_start_mask = [-100]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(["CLS"])
        for token_text, label in zip(tokens, labels):
            bpes = self.tokenizer.tokenize(token_text)
            if not bpes:
                bpes = ["[UNK]"]
            tokens_ids.extend(self.tokenizer.convert_tokens_to_ids(bpes))

            token_label_ids = [self.label_to_id[label]]
            token_start_mask_id = [self.label_to_id[label]]
            if len(bpes) > 1:
                token_label_ids.extend([self.label_to_id[label]] * (len(bpes) - 1))
                token_start_mask_id.extend([-100] * (len(bpes) - 1))

            labels_ids.extend(token_label_ids)
            token_start_mask.extend(token_start_mask_id)

        assert len(labels_ids) == len(tokens_ids), f"{labels_ids}\n{tokens_ids}"

        if len(tokens_ids) >= self.max_seq_len:
            return {
                "input_ids": torch.as_tensor(tokens_ids[:self.max_seq_len], dtype=torch.int64),
                "attention_mask": torch.as_tensor([1] * self.max_seq_len, dtype=torch.float32),
                "labels": torch.as_tensor(labels_ids[:self.max_seq_len], dtype=torch.int64),
                "token_start_mask": torch.as_tensor(token_start_mask[:self.max_seq_len], dtype=torch.int64),
            }
        pad = self.max_seq_len - len(tokens_ids)
        return {
            "input_ids": torch.as_tensor(
                tokens_ids + [self.pad_index] * pad, dtype=torch.int64
            ),
            "attention_mask": torch.as_tensor(
                [1] * len(tokens_ids) + [self.pad_index] * pad, dtype=torch.float32
            ),
            "labels": torch.as_tensor(
                labels_ids + [-100] * pad, dtype=torch.int64
            ),
            "token_start_mask": torch.as_tensor(
                token_start_mask + [-100] * pad, dtype=torch.int64
            ),
        }

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


index_to_label = {
    1: "REGION",
    2: "CURRENCY",
    3: "NORP",
    4: "MONEY",
    5: "GROUP",
    6: "EVENT",
    7: "QUANTITY",
    8: "STREET",
    9: "CITY",
    10: "NATIONALITY",
    11: "AGE",
    12: "FAMILY",
    13: "CARDINAL",
    14: "TIME",
    15: "ORGANIZATION",
    16: "ORDINAL",
    17: "LAW",
    18: "FAC",
    19: "PERSON",
    20: "COUNTRY",
    21: "GPE",
    22: "PERCENT",
    23: "LOCATION",
    24: "DATE",
    25: "PRODUCT",
    26: "PROFESSION",
    27: "WORK_OF_ART",
    28: "RELIGION",
    29: "BOROUGH",
    0: "O"
}


def transform_label_to_char_spans(dataset: CustomDataset, logits: np.array):
    """
    Метод для преобразования потокенных лейблов в спановые сущности с
    символьными координатами.
    """
    transformed_predictions = []
    assert len(dataset.tokens_per_sample) == len(logits), \
        "len(dataset.tokens_per_sample) != len(logits)"
    for sample, markup in zip(dataset.tokens_per_sample, logits):
        sample_test = [" "] * sample[-1][1]
        char_coords = []
        prev_ent = None
        for (s, e, token), label in zip(sample, markup):
            sample_test[s:e] = list(token)

            if label == "O":
                if prev_ent:
                    char_coords.append(prev_ent)
                prev_ent = None
                continue

            if prev_ent:
                if label != prev_ent[-1]:
                    char_coords.append(prev_ent)
                    prev_ent = s, e, label
                    continue
                if label == prev_ent[-1]:
                    prev_ent = prev_ent[0], e, prev_ent[-1]
                    continue

            if prev_ent is None:
                prev_ent = s, e, label
                continue
        if prev_ent:
            if label != prev_ent[-1]:
                char_coords.append(prev_ent)
                char_coords.append((s, e, label))
            if label == prev_ent[-1]:
                char_coords.append((prev_ent[0], e, prev_ent[-1]))
        transformed_predictions.append(("".join(sample_test), char_coords))
    return transformed_predictions


def cut_and_transform_predictions(
        predictions: np.array, tokens_mask: torch.tensor, index_to_label: dict
):
    """
    Обрезает паддинги входных индексов лейблов и трасформирует индексы лейблов
    в лейблы
    """
    batch_preds = []
    for pred, mask in zip(predictions, tokens_mask):
        batch_preds.append(
            [index_to_label[pred[i_m]]
             for i_m, mask_value in enumerate(mask)
             if mask_value != -100]
        )
    return batch_preds


def predict(
        model: nn.Module, predict_dl: DataLoader,
        device: str, index_to_label: dict
):
    """
    Выполняет предсказание с используя даталоадер и модель,
    а также форматирует выход модели.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in predict_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            predictions = out["logits"].argmax(dim=-1).cpu().numpy()
            preds.extend(
                cut_and_transform_predictions(
                    predictions, batch["token_start_mask"], index_to_label
                )
            )
    return preds


def inference_ner_model(
        samples: list, model: nn.Module, device: str,
        batch_size: int = 4, num_workers: int = 4,
        index_to_label: dict = index_to_label
) -> List[Tuple[str, list]]:
    dataset = CustomDataset(samples)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )
    logits = predict(model.to(device), dataloader, device, index_to_label)
    return transform_label_to_char_spans(dataset, logits)


def read_annotation_pair(ann_file: str, text_file: str):
    with open(text_file) as f:
        text = f.read()
    ents = []
    with open(ann_file) as a_f:
        anns = a_f.readlines()

    for line in anns:
        line = line.strip()
        if line:
            entity = line.split("\t")
            #  чтобы скипать разметку отношений между сущностями
            if len(entity) == 3:
                index, t_c, ent_text = entity
                try:
                    ent_type, start, end = t_c.split()
                    ent_coords = int(start), int(end), ent_type
                    ents.append(ent_coords)
                except Exception:
                    left_part, rtight_part = t_c.split(";")
                    ent_type, start, end = left_part.split()
                    ent_coords = int(start), int(end), ent_type
                    ents.append(ent_coords)
                    start, end = rtight_part.split()
                    ent_coords = int(start), int(end), ent_type
                    ents.append(ent_coords)
    return text, ents


def sentence_split(rec_text: str) -> List[Tuple[Tuple[int, int, str], list]]:
    current_pos = 0
    samples = []
    for sentence in ru_sent_tokenize(rec_text):
        shift = len(sentence) + 1
        samples.append(((current_pos, current_pos + shift, sentence), []))
        current_pos += shift
    return samples