from datetime import datetime
from typing import List


def get_date_time_rfc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def create_ok_response(
    msg_id: str,
    work_id: str,
    model_result: List[dict],
    model_time: float,
) -> dict:
    """
    Format response 200
    """
    return {
        "msgId": msg_id,
        "workId": work_id,
        "msgTm": get_date_time_rfc(),
        "modelTime": model_time,
        "modelResult": model_result
    }


def create_error_response(
    msg_id: str,
    work_id: str,
    error_msg: str = ""
) -> dict:
    """
    Format error-response because error has occurred
    """
    return {
        "msgId": msg_id,
        "workId": work_id,
        "msgTm": get_date_time_rfc(),
        "errorMsg": error_msg
    }
