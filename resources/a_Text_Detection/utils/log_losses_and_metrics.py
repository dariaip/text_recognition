from typing import Callable, Dict, Optional


def log_losses_and_metrics(
    log_method: Callable,
    losses: Dict,
    metrics: Optional[Dict],
    current_batch_size: int,
    mode: str
) -> None:
    for k, v in losses.items():
        log_method(f'{mode}_losses/{mode}_' + k, v.detach().cpu().item(), on_step=True, batch_size=current_batch_size)

    if metrics is not None:
        log_method(f'{mode}_metrics/{mode}_mean_precision_after_step', metrics['mean_precision'],
                   on_step=True, batch_size=current_batch_size)
        log_method(f'{mode}_metrics/{mode}_mean_recall_after_step', metrics['mean_recall'],
                   on_step=True, batch_size=current_batch_size)
        log_method(f'{mode}_metrics/{mode}_mean_fscore_after_step', metrics['mean_fscore'],
                   on_step=True, batch_size=current_batch_size)
        log_method(f'{mode}_metrics/{mode}_mean_support_after_step', metrics['mean_support'],
                   on_step=True, batch_size=current_batch_size)
        log_method(f'{mode}_metrics/{mode}_mean_fscore_30_after_step', metrics['fscores'][0],
                   on_step=True, batch_size=current_batch_size)
        log_method(f'{mode}_metrics/{mode}_mean_fscore_50_after_step', metrics['fscores'][4],
                   on_step=True, batch_size=current_batch_size)
        log_method(f'{mode}_metrics/{mode}_mean_fscore_75_after_step', metrics['fscores'][9],
                   on_step=True, batch_size=current_batch_size)
