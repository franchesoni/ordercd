#!/usr/bin/env python3
import argparse, csv, json, os, time, warnings
from collections import OrderedDict
from datetime import datetime, timedelta
from multiprocessing.dummy import Pool, Lock

import ee, numpy as np, rasterio, shapefile, urllib3
from rasterio.transform import Affine
from shapely.geometry import shape, Point
from skimage.exposure import rescale_intensity
from torchvision.datasets.utils import download_and_extract_archive

warnings.simplefilter('ignore', UserWarning)

# ----- Sentinel-2 & Cloud Masking -----
def maskS2clouds(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)

def get_collection(cloud_pct=20):
    return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct)) \
            .map(maskS2clouds)

def filter_collection(collection, coords, period):
    filtered = collection.filterDate(*period).filterBounds(ee.Geometry.Point(coords))
    if filtered.size().getInfo() == 0:
        raise ee.EEException(f'No images at {coords} for period {period}.')
    return filtered

# ----- Date Utilities -----
def date2str(date):
    return date.strftime('%Y-%m-%d')

def get_period(center_date, days=7):
    half = timedelta(days=days/2)
    return date2str(center_date - half), date2str(center_date + half)

# ----- Sampling Utilities -----
class GeoSampler:
    def sample_point(self):
        raise NotImplementedError()

class GaussianSampler(GeoSampler):
    def __init__(self, interest_points=None, std=50):
        if interest_points is None:
            cities = self.get_world_cities()
            self.interest_points = self.get_interest_points(cities)
        else:
            self.interest_points = interest_points
        self.std = std

    def sample_point(self):
        rng = np.random.default_rng()
        point = rng.choice(self.interest_points)
        std_deg = self.km2deg(self.std)
        return list(np.random.normal(loc=point, scale=[std_deg, std_deg]))

    @staticmethod
    def get_world_cities(download_root=os.path.expanduser('~/.cache/simplemaps')):
        url = 'https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.71.zip'
        filename = 'worldcities.csv'
        dest = os.path.join(download_root, os.path.basename(url))
        if not os.path.exists(dest):
            download_and_extract_archive(url, download_root)
        with open(os.path.join(download_root, filename)) as f:
            reader = csv.DictReader(f)
            cities = []
            for row in reader:
                row['population'] = row['population'].replace('.', '') if row['population'] else '0'
                cities.append(row)
        return cities

    @staticmethod
    def get_interest_points(cities, size=10000):
        cities = sorted(cities, key=lambda c: int(c['population']), reverse=True)[:size]
        return [[float(c['lng']), float(c['lat'])] for c in cities]

    @staticmethod
    def km2deg(kms, radius=6371):
        return kms / (2.0 * radius * np.pi / 360.0)

# ----- Raster Utilities -----
def center_crop(img, out_size):
    h, w = img.shape[:2]
    ch, cw = out_size
    top = int((h - ch + 1) * 0.5)
    left = int((w - cw + 1) * 0.5)
    return img[top:top+ch, left:left+cw]

def adjust_coords(coords, old_size, new_size):
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [coords[0][0] + ((xoff + new_size[1]) * xres), coords[0][1] - ((yoff + new_size[0]) * yres)]
    ]

def get_properties(image):
    props = {}
    for p in image.propertyNames().getInfo():
        props[p] = image.get(p)
    return ee.Dictionary(props).getInfo()

def get_patch(collection, coords, radius, bands, crop):
    image = collection.sort('system:time_start', False).first()
    region = ee.Geometry.Point(coords).buffer(radius).bounds()
    patch = image.divide(10000).select(*bands).sampleRectangle(region)
    info = patch.getInfo()
    raster = OrderedDict()
    for band in bands:
        img = np.atleast_3d(info['properties'][band])
        if crop and band in crop:
            img = center_crop(img, crop[band])
        img = rescale_intensity(img, in_range=(0, 1), out_range=np.uint8)
        raster[band] = img
    coords_arr = np.array(info['geometry']['coordinates'][0])
    new_coords = [
        [coords_arr[:, 0].min(), coords_arr[:, 1].max()],
        [coords_arr[:, 0].max(), coords_arr[:, 1].min()]
    ]
    if crop:
        b0 = bands[0]
        old_size = (len(info['properties'][b0]), len(info['properties'][b0][0]))
        new_size = raster[b0].shape[:2]
        new_coords = adjust_coords(new_coords, old_size, new_size)
    return {'raster': raster, 'coords': new_coords, 'metadata': get_properties(image)}

def save_geotiff(img, coords, filename):
    h, w, ch = img.shape
    xres = (coords[1][0] - coords[0][0]) / w
    yres = (coords[0][1] - coords[1][1]) / h
    transform = Affine.translation(coords[0][0] - xres/2, coords[0][1] + yres/2) * Affine.scale(xres, -yres)
    profile = {
        'driver': 'GTiff',
        'width': w,
        'height': h,
        'count': ch,
        'crs': '+proj=latlong',
        'transform': transform,
        'dtype': img.dtype,
        'compress': 'JPEG'
    }
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(img.transpose(2, 0, 1))

def save_patch(patch, path, preview=False, rgb_bands=('B4', 'B3', 'B2')):
    pid = patch['metadata']['system:index']
    patch_path = os.path.join(path, pid)
    os.makedirs(patch_path, exist_ok=True)
    for band, img in patch['raster'].items():
        save_geotiff(img, patch['coords'], os.path.join(patch_path, f'{band}.tif'))
    if preview:
        rgb = np.dstack([patch['raster'][b] for b in rgb_bands if b in patch['raster']])
        rgb = rescale_intensity(rgb, in_range=(0, 255*0.3), out_range=(0, 255)).astype(np.uint8)
        save_geotiff(rgb, patch['coords'], os.path.join(path, f'{pid}_preview.tif'))
    with open(os.path.join(patch_path, 'metadata.json'), 'w') as f:
        json.dump(patch['metadata'], f)

# ----- Four-Patch Acquisition -----
def get_four_patches(collection, sampler, pair_window=30, period_window=7,
                     radius=1325, bands=None, crop=None, debug=False):
    import ipdb; ipdb.set_trace()
    if bands is None:
        bands = ['B2', 'B3', 'B4']
    coords = sampler.sample_point()
    # Choose a recent base date (within past year) and an older base 8 years earlier.
    while True:
        recent_base = datetime.today() - timedelta(days=np.random.randint(0, 365))
        gap = np.random.uniform(8*365-30, 8*365+30)
        older_base = recent_base - timedelta(days=gap)
        if older_base.year >= 2015:  # Sentinel-2 data from 2015 on
            break
    delta_recent = timedelta(days=np.random.uniform(0, pair_window/2))
    delta_older = timedelta(days=np.random.uniform(0, pair_window/2))
    recent_dates = [recent_base - delta_recent, recent_base + delta_recent]
    older_dates  = [older_base - delta_older, older_base + delta_older]
    dates = older_dates + recent_dates  # older pair then recent pair
    patches = []
    for d in dates:
        period = get_period(d, days=period_window)
        try:
            col = filter_collection(collection, coords, period)
            patch = get_patch(col, coords, radius, bands, crop)
        except Exception as e:
            if debug:
                print(e)
            return get_four_patches(collection, sampler, pair_window, period_window,
                                    radius, bands, crop, debug)
        patches.append(patch)
    return {'coords': coords, 'older_pair': patches[:2], 'recent_pair': patches[2:]}

# ----- Counter -----
class Counter:
    def __init__(self, start=0):
        self.value = start
        self.lock = Lock()
    def update(self, delta=1):
        with self.lock:
            self.value += delta
            return self.value

# ----- Main -----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_locations', type=int, default=1000)
    parser.add_argument('--cloud_pct', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--pair_window', type=float, default=30, help='Days between images in a pair.')
    parser.add_argument('--period_window', type=float, default=7, help='Days span to filter images.')
    parser.add_argument('--radius', type=int, default=1325)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # set seed
    np.random.seed(args.seed)
    ee.Initialize(project="ordercd")
    collection = get_collection(cloud_pct=args.cloud_pct)
    sampler = GaussianSampler()

    ALL_BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    crop = {'B1': (44, 44), 'B2': (264, 264), 'B3': (264, 264), 'B4': (264, 264),
            'B5': (132, 132), 'B6': (132, 132), 'B7': (132, 132), 'B8': (264, 264),
            'B8A': (132, 132), 'B9': (44, 44), 'B11': (132, 132), 'B12': (132, 132)}

    counter = Counter()
    start_time = time.time()

    def worker(idx):
        patches = get_four_patches(collection, sampler,
                                   pair_window=args.pair_window,
                                   period_window=args.period_window,
                                   radius=args.radius, bands=ALL_BANDS, crop=crop,
                                   debug=args.debug)
        if args.save_path:
            loc_path = os.path.join(args.save_path, f'{idx:06d}')
            os.makedirs(loc_path, exist_ok=True)
            for pair in ('older_pair', 'recent_pair'):
                for patch in patches[pair]:
                    save_patch(patch, loc_path, preview=args.preview)
        cnt = counter.update(4)
        if cnt % args.log_freq == 0:
            print(f'{cnt} images downloaded in {time.time()-start_time:.1f}s.')

    indices = range(args.num_locations)
    if args.num_workers < 1:
        for i in indices:
            worker(i)
    else:
        with Pool(processes=args.num_workers) as pool:
            pool.map(worker, indices)
