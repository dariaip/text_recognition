from typing import List, Tuple


def filter_entities(
    samples: List[Tuple[tuple, list]], 
    needed_entities: set
):
    new_samples = []
    for index, ((s, en, text), markup) in enumerate(samples):
        markup = [i for i in markup if i[-1] in needed_entities]
        new_samples.append(((s, en, text), markup))
    return new_samples
