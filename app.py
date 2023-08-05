'''
Overall pipeline of dealing with the whole process includes:
* Text detection
* Text recognition
* Model for lines
* Joining lines into paragraphs
* NER based on the recognized text
'''

import os
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

from resources.resize import resize_aspect_ratio
from resources.dtos import Word, Line, Paragraph
from resources.bboxes import sort_boxes, sort_boxes_top2down_wrt_left2right_order, fit_bbox
from resources import paragraph_finder_model
from resources.postprocessor import Postprocessor
from resources.drawmore import draw_contours
from resources.service_utils import prepare_crops, batchings, group_words_by_lines_or_lines_by_paragraphs
from resources.resize_by_height import resize_by_height
from resources.ner_model import inference_ner_model


# Constants
MODEL_PATH = r'.\pretrained_models'
device = 'cpu'
max_image_size = 2048
size = 512
_transform = A.Compose([
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255.,
        p=1.,
    ),
    ToTensorV2(p=1.)
])
_postprocessor = Postprocessor(
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
        Class for transforming a text into a tensor for putting it into CTCLoss and transforming a tensor into text.

        Args:
            tokens: list of tokens (symbols) considered as an alphabet
        """
        self.tokens = tokens
        self.blank_token = 0
        self.n_tokens = len(tokens) + 1
        self.char_idx = {}

        for i, char in enumerate(tokens):
          self.char_idx[char] = i + 1

    def encode(self, text: str) -> Tuple[torch.Tensor, int]:
        """
        Method for transforming a text into a tensor for CTCLoss.

        Args:
            text: a text

        Returns:
            tokenized test in form of a tensor, length of incoming text
        """
        result = []
        for char in text:
          result.append([self.char_idx[char]])

        return torch.LongTensor(result), len(result)

    def decode(self, preds: List[Any]) -> str:
        """
        Method for transorming a tokenized text into a normal text

        Args:
            preds: model's predictions for one line of text

        Returns:
            model's confidence in form of probability; a line of the recognized text
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
    # Split the provided image using the method "batchings"
    for batch in batchings(image, batch_size):
        # Transform each image of the batch using "resize_by_height"
        batch = [resize_by_height(img, target_height) for img in batch]
        # Calculate maximum width of an image in the batch
        max_width = np.max([img.shape[1] for img in batch])
        # Add paddings to all images in the batch in order to get the same widths of images
        batch = [cv2.copyMakeBorder(
            img,
            0, 0,
            0,
            (max_width - img.shape[1] + pad_value),
            cv2.BORDER_CONSTANT
        )
        for img in batch]
        # preprocess an image
        batch_changed = [
            transform(image=img)['image']
            for img in batch
        ]
        # Transform all images into tensors and join them into the only tensor using torch.stack
        batch = torch.stack([torch.Tensor(img) for img in batch_changed])
        # get model's predictions
        with torch.no_grad():
            preds = model(batch).to(device)
        # postprocess predictions
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
    # preprocess images
    h, w, _ = image.shape
    image_changed = transform(image=image)['image']
    image_changed = torch.Tensor(image_changed[np.newaxis,:])
    # get model's predictions
    with torch.no_grad():
        prediction = model(image_changed.to(device))
    # postprocess predictions
    pred_image = prediction[0].cpu().detach().numpy()
    bboxs = postprocessor(w, h, pred_image, return_polygon=False)[0][0]
    # нормализация bounding box'ов по высоте и ширине
    bboxs_arrays = np.array([np.array(x) for x in bboxs])
    normalized_bboxs=bboxs_arrays/np.array([w, h])
    # create a list of objects of type "Line"
    res = [Line(bbox=x, normalized_bbox=y) for x, y in zip(bboxs_arrays, normalized_bboxs)]

    return res


@st.cache_data
def detect_text(uploaded_file, max_image_size, _transform, _postprocessor, device='cpu', thickness=2, color=(0, 0, 255)):
    image = np.asarray(bytearray(uploaded_file.read()), np.uint8)
    image = cv2.imdecode(image, 1)
    image_resized, _, _ = resize_aspect_ratio(image, square_size=max_image_size, interpolation=cv2.INTER_LINEAR)
    model_fpath = os.path.join(MODEL_PATH, 'model_td.jit')
    text_detection_model = torch.jit.load(model_fpath, map_location=torch.device(device))
    text_detection_model.eval()
    pred_bboxes = text_detection_inference(text_detection_model, image_resized, _transform, _postprocessor, device)
    countours_result = draw_contours(image_resized, pred_bboxes, thickness=thickness, color=color)
    return countours_result, image_resized, pred_bboxes, image


@st.cache_resource
def recognize_text(image_resized, pred_bboxes, _transform, _tokenizer, device='cpu', batch_size=8):
    crops = prepare_crops(image_resized, pred_bboxes)
    ocr_model_fpath = os.path.join(MODEL_PATH, 'model_tr.jit')
    ocr_model = torch.jit.load(ocr_model_fpath, map_location=torch.device(device))
    ocr_model.eval()
    labels = ocr_inference(ocr_model, crops, _transform, _tokenizer, device, batch_size=batch_size)
    words = [Word(bbox, label) for bbox, label in zip(pred_bboxes, labels) if len(label) > 0]
    return words, ocr_model


@st.cache_resource
def get_lines(words, image, image_resized, _transform, _tokenizer, _postprocessor, _ocr_model,
              device='cpu', sorting_type='left2right', thickness=2):
    line_model_path = os.path.join(MODEL_PATH, 'la.jit')
    line_model = torch.jit.load(line_model_path, map_location=torch.device(device))
    line_model.eval()
    lines = line_detector_inference(line_model, image_resized, _transform, _postprocessor, device)
    lines = sort_boxes(lines)

    h, w, _ = image_resized.shape
    lines = group_words_by_lines_or_lines_by_paragraphs(words, lines, w, h)
    lines = [line for line in lines if len(line.items) > 0]
    for line in lines:
        line.items = sort_boxes(line.items, sorting_type=sorting_type)
        line.label = ' '.join([word.label.strip() for word in line.items])
    lines_result = draw_contours(image_resized, [line.bbox for line in lines], thickness=thickness)
    pred_bboxes = [line.bbox for line in lines]
    line_crops = prepare_crops(image, pred_bboxes)
    line_labels = ocr_inference(_ocr_model, line_crops, _transform, _tokenizer, device, batch_size=8)
    for i, (line, line_label) in enumerate(zip(lines, line_labels)):
        line.label = line_label
    return lines_result, line_labels, lines


def get_layout(lines, image_resized, overlapping_threshold, cluster_eps, thickness=2):
    paragraph_finder = paragraph_finder_model.ParagraphFinder(
        overlapping_threshold=overlapping_threshold,
        cluster_eps=10 ** cluster_eps
    )
    paragraphs = paragraph_finder.find_paragraphs(lines)
    for para in paragraphs:
        para.label = ' '.join([line.label.strip() for line in para.items])
    para_result = draw_contours(image_resized, [para.bbox for para in paragraphs], thickness=thickness)
    return para_result, paragraphs







if __name__ == '__main__':
    st.title("Demonstration of the Text Recognition pipeline step by step")
    uploaded_file = st.file_uploader("Upload your image here...")
    if uploaded_file:
        # upload an image
        st.subheader("Step 1 of 7")
        st.write("The initial image looks like this. If there is text on the image, it should be recognized at the end of the pipeline.")
        expander = st.expander("Uploaded image", expanded=True)
        expander.image(uploaded_file)

        # text detection
        st.divider()
        st.subheader("Step 2 of 7")
        st.write("First of all, we need to detect areas on the image which include text.")
        countours_result, image_resized, pred_bboxes, image = detect_text(
            uploaded_file,
            max_image_size,
            _transform,
            _postprocessor,
            device=device,
            thickness=2,
            color=(0, 0, 255)
        )
        expander = st.expander("Detected bounding boxes with text")
        expander.image(countours_result.astype('int'))

        # Text recognition
        alphabet = punct + digit + cr + latin
        _tokenizer = TokenizerForCTC(list(alphabet))
        words, _ocr_model = recognize_text(image_resized, pred_bboxes, _transform, _tokenizer, device=device, batch_size=8)

        # Joining words into lines and paragraphs
        st.divider()
        st.subheader("Step 3 of 7")
        st.write("The bounding boxes ahead include separate letters/symbols or words. But when we're dealing with documents (plenty of letters), we're interested in determining its structure because it adds some meaning to the text. At this stage, we determine a simple level of a text's structure - lines.")
        lines_result, line_labels, lines = get_lines(
            words,
            image,
            image_resized,
            _transform,
            _tokenizer,
            _postprocessor,
            _ocr_model,
            device=device,
            sorting_type='left2right',
            thickness=2
        )
        expander = st.expander("Detected text grouped in lines")
        expander.image(lines_result.astype('int'))

        st.divider()
        st.subheader("Step 4 of 7")
        st.write("Let's look at the recognized text. Maybe your image is simple enough to stop here.")
        expander = st.expander("Recognized text (line by line)")
        expander.text_area('Text', '\n'.join(line_labels), height=500)

        # join into paragraphs
        st.divider()
        st.subheader("Step 5 of 7")
        st.write("As you can see, understanding lines are not enough because stopping at this step, we can find ourselves in a situation where the order of words/lines can be wrong because of a more complex structure presented in the document's layout. That's why we need to recognize the layout (paragraphs/columns/etc.).")
        expander = st.expander("Detected text grouped in paragraphs")
        overlapping_threshold = expander.slider('overlapping_threshold:', 0., 1., 0.5)
        cluster_eps = expander.slider('cluster_eps (hyperparameter of DBSCAN, power of 10):', -5, 5, -1)
        if 0 <= overlapping_threshold <= 1 and -5 <= cluster_eps <= 5:
            para_result, paragraphs = get_layout(lines, image_resized, overlapping_threshold, cluster_eps, thickness=2)
            expander.image(para_result.astype('int'))

            # text from paragraphs
            st.divider()
            st.subheader("Step 6 of 7")
            st.write("And finally, we move to the characters' recognition and join them into a text following the correct order (according to the determined layout).")
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
            st.subheader("Step 7 of 7")
            st.write("When we have a text we can use NLP techniques for its analysis. For example, let's extract such standard entities as *Person* or *Organization*")
            model = torch.jit.load(os.path.join(MODEL_PATH, "ner_rured.jit"))
            model.eval()
            # Prepare examples for dataset in the same format that was used during the model's training
            samples = [((0, len(rec_text), rec_text), [])]
            result = inference_ner_model(samples, model, device, batch_size=1, num_workers=1)
            with st.expander("Extracted named entities", expanded=True):
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


