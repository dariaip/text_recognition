from typing import Optional


def make_label(predicted_text: Optional[str], target_text: Optional[str]) -> str:
    """
    Вспомогательная функция для создания подписи с истинным / предсказанным текстом для отображения на графике.

    Args:
        predicted_text: предсказанный моделью текст (опционально)
        target_text: истинный текст (опционально)

    Returns:
        подпись для отображения на графике
    """
    label = ''
    if predicted_text is not None:
        label += f'Pred: {predicted_text}'
    if target_text is not None:
        label += f'\nTrue: {target_text}'
    return label.strip().replace('$', '\\$')
