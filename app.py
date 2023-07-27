'''
Main pipeline of dealing with the whole process includes:
* Text detection
* Text recognition
* Model for lines
* Joining lines into paragraphs
* Four previous steps as a method of an image processing
* NER based on the recognized text
* Data processing and end-to-end metrics' calculation
* Service using flask (or streamlit?)
'''

import os
import sys
from typing import Union, List, Any, Tuple
import numpy as np
from itertools import groupby

import cv2
import torch
import torch.nn as nn
import albumentations as A
from albumentations import BasicTransform, Compose, OneOf
from albumentations.pytorch import ToTensorV2

import streamlit as st
from annotated_text import annotated_text


sys.path.append(r'.\resources')

from c_Layout_Analisys.utils import (resize_aspect_ratio, Word, Line, sort_boxes,
                                     sort_boxes_top2down_wrt_left2right_order, fit_bbox, Paragraph)
from c_Layout_Analisys import paragraph_finder_model
from a_Text_Detection.utils import Postprocessor, DrawMore
from e_Service_Deployment.utils import prepare_crops, batchings, group_words_by_lines_or_lines_by_paragraphs
from b_Optical_Character_Recognition.utils import resize_by_height
from e_Service_Deployment.ner_model import inference_ner_model


# Constants
MODEL_PATH = r'.\pretrained_models'
device = 'cpu'
max_image_size = 2048
index2label = {
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
size = 512
transform = A.Compose([
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255.,
        p=1.,
    ),
    ToTensorV2(p=1.)
])
postprocessor = Postprocessor(
    unclip_ratio=1.5,
    binarization_threshold=0.3,
    confidence_threshold=0.7,
    min_area=1,
    max_number=1000
)
punct = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~«»№"
digit = "0123456789"
cr = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё"
latin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def text_detection_inference(
    model: nn.Module,
    image: np.ndarray,
    transform: Union[BasicTransform, Compose, OneOf],
    postprocessor: Postprocessor,
    device: str = 'cpu',
) -> List[np.ndarray]:
    # image preparation
    h, w, c = image.shape
    image_changed = transform(image=image)['image'].unsqueeze(0).to(device)
    # making predictions
    with torch.no_grad():
        prediction = model(image_changed).to(device)
    # prediction's postprocessing
    pred_bin_maps = prediction[:, [2], :, :].cpu().detach().numpy()
    return [np.array(x) for x in postprocessor(w, h, pred_bin_maps, return_polygon=False)[0]][0]


class TokenizerForCTC:
    def __init__(self, tokens: List[Any]):
        """
        Класс для преобразования текста в тензор для подачи в CTCLoss и преобразования из тензора в текст.

        Args:
            tokens: список токенов (символов) из алфавита
        """
        self.tokens = tokens
        self.blank_token = 0
        self.n_tokens = len(tokens) + 1
        self.char_idx = {}

        for i, char in enumerate(tokens):
          self.char_idx[char] = i + 1

    def encode(self, text: str) -> Tuple[torch.Tensor, int]:
        """
        Метод для преобразования текста в тензор для CTCLoss.

        Args:
            text: текст

        Returns:
            токенизированный текст в виде тензора, длина входного текста
        """
        result = []
        for char in text:
          result.append([self.char_idx[char]])

        return torch.LongTensor(result), len(result)

    def decode(self, preds: List[Any]) -> str:
        """
        Метод для перевода токенизированного текста в обычный текст. #CTC approach

        Args:
            preds: предсказания модели с одной строкой текста

        Returns:
            уверенность модели в виде вероятностей; строка с распознанным текстом
        """
        remove_duplicates = [g for g, _ in groupby(preds)]
        remove_zeros = [g for g in remove_duplicates if g != 0]
        return ''.join([self.tokens[g-1] for g in remove_zeros])


def ocr_inference(
    model: nn.Module,
    image: np.ndarray,
    transform: Union[BasicTransform, Compose, OneOf],
    tokenizer: TokenizerForCTC,
    device: str = 'cpu',
    batch_size: int = 1,
    target_height: int = 32,
    pad_value: int = 0
) -> List[str]:
    strings = []
    # Разбить входящие изображения с помощью метода batchings
    for batch in batchings(image, batch_size):
        # Каждую картинку в батче преобразовать с помощью resize_by_height
        batch = [resize_by_height(img, target_height) for img in batch]
        # Вычислить максимальную ширину изображения в батче
        max_width = np.max([img.shape[1] for img in batch])
        # Добить все изображения в батче до одной ширины значениями pad_value
        batch = [cv2.copyMakeBorder(
            img,
            0, 0,
            0,
            (max_width - img.shape[1] + pad_value),
            cv2.BORDER_CONSTANT
        )
        for img in batch]
        # подготовка изображения
        batch_changed = [
            transform(image=img)['image']
            for img in batch
        ]
        # Привести все изображения к тензорам и объединить в один тензор через torch.stack
        batch = torch.stack([torch.Tensor(img) for img in batch_changed])
        # предсказание модели (с помощью model)
        with torch.no_grad():
            preds = model(batch).to(device)
        # постпроцессинг предсказаний
        preds = preds.log_softmax(dim=2)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).detach().cpu().numpy()
        pred_labels = [tokenizer.decode(p) for p in preds]
        string_in_batch = [''.join(pl) for pl in pred_labels]
        strings += string_in_batch
    return strings


def line_detector_inference(
    model: nn.Module,
    image: np.ndarray,
    transform: Union[BasicTransform, Compose, OneOf],
    postprocessor: Postprocessor,
    device: str = 'cpu',
) -> List[Line]:
    # подготовка изображения (c помощью preprocessor)
    h, w, _ = image.shape
    image_changed = transform(image=image)['image']
    image_changed = torch.Tensor(image_changed[np.newaxis,:])
    # предсказание модели (с помощью model)
    with torch.no_grad():
        prediction = model(image_changed.to(device))
    # постпроцессинг предсказаний (с помощью postprocessor)
    pred_image = prediction[0].cpu().detach().numpy()
    bboxs = postprocessor(w, h, pred_image, return_polygon=False)[0][0]
    # нормализация bounding box'ов по высоте и ширине
    bboxs_arrays = np.array([np.array(x) for x in bboxs])
    normalized_bboxs=bboxs_arrays/np.array([w, h])
    # создание списка объектов типа Line
    res = [Line(bbox=x, normalized_bbox=y) for x, y in zip(bboxs_arrays, normalized_bboxs)]

    return res




# upload an image
st.title("Demonstration of the Text Recognition pipeline step by step")
uploaded_file = st.file_uploader("Upload your image here...")
if uploaded_file:
    st.subheader("The initial image looks like this. If there is a text on it, the text should be recognized to the end of pipeline.")
    expander = st.expander("Uploaded image")
    expander.image(uploaded_file)

    # text detection
    st.divider()
    st.subheader("First of all, we need to detect areas on the image which include text.")
    image = np.asarray(bytearray(uploaded_file.read()), np.uint8)
    image = cv2.imdecode(image, 1)
    image_resized, _, _ = resize_aspect_ratio(image, square_size=max_image_size, interpolation=cv2.INTER_LINEAR)
    model_fpath = os.path.join(MODEL_PATH, 'model_td.jit')
    text_detection_model = torch.jit.load(model_fpath, map_location=torch.device(device))
    text_detection_model.eval()
    pred_bboxes = text_detection_inference(text_detection_model, image_resized, transform, postprocessor, device)
    countours_result = DrawMore.draw_contours(image_resized, pred_bboxes, thickness=2, color=(0, 0, 255))
    expander = st.expander("Detected bounding boxes with text")
    expander.image(countours_result.astype('int'))

    # Text recognition
    crops = prepare_crops(image_resized, pred_bboxes)
    alphabet = punct + digit + cr + latin
    tokenizer = TokenizerForCTC(list(alphabet))
    ocr_model_fpath = os.path.join(MODEL_PATH, 'model_tr.jit')
    ocr_model = torch.jit.load(ocr_model_fpath, map_location=torch.device(device))
    ocr_model.eval()
    labels = ocr_inference(ocr_model, crops, transform, tokenizer, device, batch_size=8)
    words = [Word(bbox, label) for bbox, label in zip(pred_bboxes, labels) if len(label) > 0]

    # Joining words into lines and paragraphs
    st.divider()
    st.subheader("The bounding boxes ahead include separate letters/symbols or words. But when we're dealing with documents (plenty of letters), we're interested in determining its structure, because it adds some meaning to the text. On this stage we determine a simple level of a text's structure - lines.")
    line_model_path = os.path.join(MODEL_PATH, 'la.jit')
    line_model = torch.jit.load(line_model_path, map_location=torch.device(device))
    line_model.eval()
    lines = line_detector_inference(line_model, image_resized, transform, postprocessor, device)
    lines = sort_boxes(lines)

    h, w, _ = image_resized.shape
    lines = group_words_by_lines_or_lines_by_paragraphs(words, lines, w, h)
    lines = [line for line in lines if len(line.items) > 0]
    for line in lines:
        line.items = sort_boxes(line.items, sorting_type='left2right')  # сортировка слева направо
        line.label = ' '.join([word.label.strip() for word in line.items])
    lines_result = DrawMore.draw_contours(image_resized, [line.bbox for line in lines], thickness=2)
    expander = st.expander("Detected text grouped in lines")
    expander.image(lines_result.astype('int'))

    pred_bboxes = [line.bbox for line in lines]
    line_crops = prepare_crops(image, pred_bboxes)
    line_labels = ocr_inference(ocr_model, line_crops, transform, tokenizer, device, batch_size=8)
    st.divider()
    st.subheader("Let's look at the recognized text. Maybe your image is simple enough to stop here.")
    expander = st.expander("Recognized text (line by line)")
    for i, (line, line_label) in enumerate(zip(lines, line_labels)):
        line.label = line_label
    expander.text_area('Text', '\n'.join(line_labels), height=500)

    # join into paragraphs
    st.divider()
    st.subheader("As you can see, understanding lines is not enough, because stopping on this step, we can find ourselves in a situation when order of words/lines can be wrong because of a more complex structure presented in the document's layout. That's why we need to recognize the layout (paragraphs / columns / etc.).")
    expander = st.expander("Detected text grouped in paragraphs")
    overlapping_threshold = expander.slider('overlapping_threshold:', 0., 1., 0.5)
    cluster_eps = expander.slider('cluster_eps (hyperparameter of DBSCAN, power of 10):', -5, 5, -1)
    if 0 <= overlapping_threshold <= 1 and -5 <= cluster_eps <= 5:
        paragraph_finder = paragraph_finder_model.ParagraphFinder(
            overlapping_threshold=overlapping_threshold,
            cluster_eps=10**cluster_eps
        )
        paragraphs = paragraph_finder.find_paragraphs(lines)
        for para in paragraphs:
            para.label = ' '.join([line.label.strip() for line in para.items])
        para_result = DrawMore.draw_contours(image_resized, [para.bbox for para in paragraphs], thickness=2)
        expander.image(para_result.astype('int'))

        # text from paragraphs
        st.divider()
        st.subheader("And finally we move to the characters' recognition and joining them into a text following the right order (according to the determined layout).")
        expander = st.expander("Recognized text (paragraph by paragraph, line by line)")
        rec_text = ''
        for i, para in enumerate(paragraphs):
            line_labels = [line.label for line in para.items]
            rec_text += ' ' + ' '.join(line_labels)
            expander.text_area(
                f'Paragraph {i+1}',
                '\n'.join(line_labels),
                height=min(500, len(line_labels)*10)
            )
        rec_text = rec_text.strip()

        # NER
        st.divider()
        st.subheader("When we have a text, we can use NLP techniques for its analisys. For example, let's extract such standard entities as *Person* or *Organization*")
        model = torch.jit.load(os.path.join(MODEL_PATH, "ner_rured.jit"))
        model.eval()
        # Готовим примеры для подачи в датасет, оставляем формат, который был использован при обучении, но без сущностей
        samples = [((0, len(rec_text), rec_text), [])]
        result = inference_ner_model(samples, model, device, index_to_label=index2label, batch_size=1, num_workers=1)
        with st.expander("Extracted named entities"):
            for t, ents in result:
                t = [(t, (0, len(t)), None)]
                tmp = []
                for s, e, ent_type in ents:
                    for i, tt in enumerate(t):
                        if tt[1][0] <= s and tt[1][1] >= e and tt[2] is None:
                            local_text, (start, end), et = tt
                            tmp.append((local_text[0:s-start], (start, s), None))
                            tmp.append((local_text[s-start:e-start], (s, e), ent_type))
                            tmp.append((local_text[e-start:end-start], (e, end), None))
                        else:
                            tmp.append(tt)
                    t, tmp = tmp, []
                annotated_text(*[tt[0] if tt[2] is None else (tt[0], tt[2]) for tt in t])


