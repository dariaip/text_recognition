### Description

The repository is a result of doing tasks during patricipation into **Middle DS School: Text Recognition** course.

The trained models are related to processing non-structured documents, including such tasks as Text Detection, Optical Character Recognition, Layout Analysis and Named Entity Recognition. 
The task can be referenced to **CV**, **NLP** fields as well as Multimodal approaches which process text and image simultaneously. мультимодальные подходы позволяют полностью.

The course also included dealing with such tools for **MLOPs** process as: **Docker**,
**W&B**, frameworks **onnx** and **openvino**. The resulting product is an end-to-end service of image processing, that you can test.

### Demo (speeded-up artificially)

https://github.com/dariaip/text_recognition/assets/41380940/76a9a054-d578-4613-8dff-180f2b0a5d7a




### How to run the service 
python -m venv venv *(I use python 3.10)*

source venv/bin/activate *(or venv\Scripts\activate in case of Windows)*

python -m pip install --upgrade pip

python -m pip install --upgrade setuptools

python -m pip install -r requirements.txt

streamlit run app.py
