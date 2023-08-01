import json
from time import time

import numpy as np
from flask import Flask, jsonify, make_response, request

from service_utils import create_ok_response, create_error_response

COMMON_VERSION = "0.0.1"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000


class Pipeline:

    def __init__(self):
        pass

    def predict(self, image) -> dict:
        return {
            "recognized_text": "Изображение получили и даже распознали",
            "entities": [
                {"name": "Единственная сущность", "coords": [0, 5], "text": "Неопределен"}
            ]
        }


def create_app():
    try:
        pipe = Pipeline()

    except Exception as e:
        raise Exception(f"Can not load Pipeline: {e}")

    app = Flask(__name__)

    @app.route("/version", methods=["GET"])
    def version():
        version_data = {
            "common": COMMON_VERSION
        }
        return make_response(jsonify({"version": version_data}), 200)

    @app.route("/health", methods=["GET"])
    def health():
        output_data = {
            "health_status": "running"
        }
        return make_response(jsonify(output_data), 200)

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        params: {
            "msgId":     <unique message id>
            "workId":    <unique request id, will be returned in answer>
            "msgTm":     <message time in format: %Y-%m-%dT%H:%M:%S.%fZ>
            "image":     <image for recognition and extraction>
        }
        """

        received_image = request.files.get("image")
        if not received_image:
            return make_response(jsonify({
                "errorMsg": "No file with key \"image\" was found"
            }), 400)

        image_bytes = received_image.read()

        req_params = request.form.get("requestParameters")
        if not req_params:
            return make_response(jsonify({
                "errorMsg": "Expected key \"requestParameters\", but not found"
            }), 400)

        input_params = json.loads(req_params)

        for param in ["msgId", "workId", "msgTm"]:
            if param not in input_params:
                return make_response(jsonify({
                    "errorMsg": f"Form key requestParameters/\"{param}\" is not set!"
                }), 400)

        try:

            t_start = time()

            image = np.fromstring(image_bytes, np.uint8)

            model_result = pipe.predict(image)

        except Exception as e:
            output_data = create_error_response(
                msg_id=input_params["msgId"],
                work_id=input_params["workId"],
                error_msg=str(e)
            )
            return make_response(jsonify(output_data), 500)

        t_end = time()

        output_data = create_ok_response(
            msg_id=input_params["msgId"],
            work_id=input_params["workId"],
            model_result=model_result,
            model_time=t_end - t_start
        )

        return make_response(json.dumps(output_data, ensure_ascii=False), 200)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, threaded=False)
