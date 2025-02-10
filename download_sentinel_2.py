import tqdm
import time
import numpy as np
from datasets import load_dataset as hf_load_dataset
import os
from hftoken import token_value

DATASET = "satellogic/EarthView"
sets = {
    "satellogic": {
        "shards": 7863,
    },
    "sentinel_1": {
        "shards": 1763,
    },
    "neon": {
        "config": "default",
        "shards": 607,
        "path": "data",
    },
    "sentinel_2": {
        "shards": 19997,
    },
}


def main():
    os.environ["HF_TOKEN"] = token_value

    ds = hf_load_dataset(
        path="satellogic/EarthView",
        name="sentinel_2",
        save_infos=True,
        split="train",
        data_files=None,
        streaming=True,
        token=os.environ.get("HF_TOKEN", None),
    )

    print("downloading dataset...")
    st = time.time()
    os.makedirs("data/sentinel_2", exist_ok=True)
    for i, sample in tqdm.tqdm(enumerate(ds)):
        np.save(f"data/sentinel_2/{i}.npy", np.array(sample["rgb"], dtype=np.uint8))
    print("finished downloading dataset")
    print(f"elapsed time: {time.time() - st}")


if __name__ == "__main__":
    main()
