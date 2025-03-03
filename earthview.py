import time
import tqdm
from datasets import load_dataset as hf_load_dataset
from os import environ
from PIL import Image
import numpy as np
import json

from pyarrow.parquet import ParquetFile
from pyarrow import Table as pa_Table
from datasets import Dataset

DATASET = "satellogic/EarthView"

sets = {
    "satellogic": {
        "shards" : 7863,
    },
    "sentinel_1": {
        "shards" : 1763,
    },
    "neon": {
        "config" : "default",
        "shards" : 607,
        "path"   : "data",
    },
    "sentinel_2": {
        "shards" : 19997,
    },
}

def get_subsets():
    return sets.keys()

def get_nshards(subset):
    return sets[subset]["shards"]

def get_path(subset):
    return sets[subset].get("path", subset)

def get_config(subset):
    return sets[subset].get("config", subset)

def load_dataset(subset, dataset="satellogic/EarthView", split="train", shards = None, streaming=True, **kwargs):
    config = get_config(subset)
    nshards = get_nshards(subset)
    path   = get_path(subset)
    if shards is None:
        data_files = None
    else:
        if subset == "sentinel_2":
            data_files = [f"{path}/sentinel_2-{shard//10}/{split}-{shard % 10:05d}-of-00010.parquet" for shard in shards]
        else:
            data_files = [f"{path}/{split}-{shard:05d}-of-{nshards:05d}.parquet" for shard in shards]
        data_files = {split: data_files}

    ds = hf_load_dataset(
        path=dataset,
        name=config,
        save_infos=True,
        split=split,
        data_files=data_files,
        streaming=streaming,
        token=environ.get("HF_TOKEN", None),
        **kwargs)

    return ds    

def load_parquet(subset_or_filename, batch_size=100):
    if subset_or_filename in get_subsets():
        filename = f"dataset/{subset_or_filename}/sample.parquet"
    else:
        filename = subset_or_filename

    pqfile = ParquetFile(filename)
    batch  = pqfile.iter_batches(batch_size=batch_size)
    return Dataset(pa_Table.from_batches(batch))

def item_to_images(subset, item):
    """
    Converts the images within an item (arrays), as retrieved from the dataset to proper PIL.Image

    subset: The name of the Subset, one of "satellogic", "neon", "sentinel-1"
    item: The item as retrieved from the subset

    returns the item, with arrays converted to PIL.Image
    """
    metadata = item["metadata"]
    if type(metadata) == str:
        metadata = json.loads(metadata)

    item = {
        k: np.asarray(v).astype("uint8")
            for k,v in item.items()
                if k != "metadata"
    }
    item["metadata"] = metadata
    
    if subset == "satellogic":
        # item["rgb"] = [
        #     Image.fromarray(np.average(image.transpose(1,2,0), 2).astype("uint8"))
        #         for image in item["rgb"]
        # ]
        rgbs = []
        for rgb in item["rgb"]:
            rgbs.append(Image.fromarray(rgb.transpose(1,2,0)))
            # rgbs.append(Image.fromarray(rgb[0,:,:]))      # Red
            # rgbs.append(Image.fromarray(rgb[1,:,:]))      # Green
            # rgbs.append(Image.fromarray(rgb[2,:,:]))      # Blue
        item["rgb"] = rgbs
        item["1m"] = [
            Image.fromarray(image[0,:,:])
                for image in item["1m"]
        ]
        count = len(item["1m"])
    elif subset == "sentinel_1":
        # Mapping of V and H to RGB. May not be correct
        # https://gis.stackexchange.com/questions/400726/creating-composite-rgb-images-from-sentinel-1-channels
        i10m = item["10m"]
        i10m = np.concatenate(
            (   i10m,
                np.expand_dims(
                    i10m[:,0,:,:]/(i10m[:,1,:,:]+0.01)*256,
                    1
                ).astype("uint8")
            ),
            1
        )
        item["10m"] = [
            Image.fromarray(image.transpose(1,2,0))
                for image in i10m
        ]
        count = len(item["10m"])
    elif subset == "sentinel_2":
        for channel in ['10m', '20m', 'rgb', 'scl']: #, '40m']:
            data = item[channel]
            count = len(data)
            data = np.asarray(data).astype("uint8").transpose(0,2,3,1)
            if channel == "20m":
                data = data[:,:,:,[0,2,4]]
            mode = "L" if channel in ["10m", "scl"] else "RGB"
            images = [Image.fromarray(data[i].squeeze(), mode=mode) for i in range(count)]
            item[channel] = images
        for field in ["solarAngles", "tileGeometry", "viewIncidenceAngles"]:
            item["metadata"][field] = [json.loads(s) for s in item["metadata"][field]]
    elif subset == "neon":
        item["rgb"] = [
            Image.fromarray(image.transpose(1,2,0))
                for image in item["rgb"]
        ]
        item["chm"] = [
            Image.fromarray(image[0])
                for image in item["chm"]
        ]

        # The next is a very arbitrary conversion from the 369 hyperspectral data to RGB
        # It just averages each 1/3 of the bads and assigns it to a channel
        item["1m"] = [
            Image.fromarray(
                np.concatenate((
                    np.expand_dims(np.average(image[:124],0),2),
                    np.expand_dims(np.average(image[124:247],0),2),
                    np.expand_dims(np.average(image[247:],0),2))
                ,2).astype("uint8"))
                    for image in item["1m"]
        ]
        count = len(item["rgb"])
        bounds = item["metadata"]["bounds"]

        # swap pairs
        item["metadata"]["bounds"] = [bounds[i+1-l] for i in range(0, len(bounds), 2) for l in range(2)]

        # fix CRS
        item["metadata"]["epsg"] = "EPSG:4326"
    
    item["metadata"]["count"] = count
    return item


if __name__ == '__main__':
    # download sentinel_2 imagery
    shards = list(range(10))
    from hftoken import token_value
    environ["HF_TOKEN"] = token_value
    print('downloading dataset...')
    st = time.time()
    data = load_dataset("sentinel_2", split="train", streaming=False, shards=shards)
    print(f'downloaded {len(shards)} shards in', time.time()-st, 'seconds')

    # multidate = 0
    # n_images = 0
    # st = time.time()
    # for i, item in tqdm.tqdm(enumerate(data)):
    #     n_dates = len(item['metadata']['timestamp']) 
    #     if n_dates > 1:
    #         multidate += 1
    #         n_images += len(item['metadata']['timestamp'])
    #     print(i+1, f'samples at {(i+1)/(st-time.time())} samples/sec, of which', multidate, f'are multidate ({multidate / (i+1):.2f}%), with mean of {n_images/(i+1):.2f} imgs/sample' + ' '*30, end='\r')
    # print('final count:', i+1, multidate, multidate / (i+1))
    # print('iterated in', time.time()-st, 'seconds')