from tqdm.notebook import tqdm
from glob import glob


def read_annotation(dataset_path: str, dataset_name: str):
  data = []
  for file in tqdm(glob(f"{dataset_path}/*.txt"), desc=f"Read {dataset_name} data..."):
    with open(file) as f:
      text = f.read()
    ann_file = f"{file[:-4]}.ann"
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
    data.append((text.strip(), ents))
  return data
