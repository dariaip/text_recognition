### Description

The repository is a result of participation in the in-house **Text Recognition** course. 

The trained models are related to processing non-structured documents, including such tasks as Text Detection, Optical Character Recognition, Layout Analysis, and Named Entity Recognition. 
The task can be referenced to **CV**, **NLP** fields.

The course also covered dealing with such tools for **MLOPs** process as: **Docker**, **W&B**, frameworks **onnx**, and **openvino**. The resulting product is an end-to-end service of image processing, that you can test.

My impact into doing the project includes:
- Training and finetuning all models (including writing code for it)
- Creating the provided demonstration from scratch
 
Pay attention that the pipeline is focused on processing Cyrillic symbols with some injections of Latin letters which is a widespread situation for formal documents in Russia.



### Demo (speeded-up artificially)

https://github.com/dariaip/text_recognition/assets/41380940/76a9a054-d578-4613-8dff-180f2b0a5d7a




### How to run the service 

```
# I use python 3.10
python -m venv venv
source venv/bin/activate
# or venv\Scripts\activate in case of Windows
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
python -m pip install -r requirements.txt
streamlit run app.py
```
