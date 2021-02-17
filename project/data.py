"""
This implements all data collection tasks, i.e. it pulls images from sentinel hub
services and stores the data as well as the NDVI version of it in the respective
data folder.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import config

from sentinelhub import SHConfig
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataSource, bbox_to_dimensions, DownloadRequest
from PIL import Image
from datetime import datetime, timedelta


def daterange(start_date, end_date):
    """
    Date range generator function.

    :param start_date:  start date of date range
    :param end_data:    end date of date range
    """
    for n in range(0, int((end_date - start_date).days), config.ORBITAL_PERIOD):
        yield start_date + timedelta(n)


def authorize():
    """
    Set cridentials for sentinel hub services.

    :return:    sentinel hub service configurations
    """
    sh_config = SHConfig()

    if config.CLIENT_ID and config.CLIENT_SECRET:
        sh_config.sh_client_id = config.CLIENT_ID
        sh_config.sh_client_secret = config.CLIENT_SECRET

    if sh_config.sh_client_id == '' or sh_config.sh_client_secret == '':
        print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")

    return sh_config


def pull_images(coords):
    """
    Pulls satellite images with the respective parameters from sentinel hub services
    and stores them in the data folder.

    :param coords:  coordinates of RoI
    """

    bbox = BBox(bbox=coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=config.RES)

    # create data directory to store pulled data
    data_path = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    print(f'Image shape at {config.RES} m resolution: {size} pixels')

    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04", "B08"]
                }],
                output: {
                    bands: 4
                }
            };
        }

        function evaluatePixel(sample) {

            //return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02];
            return [sample.B02, sample.B03, sample.B04, sample.B08];
        }
    """

    for year_idx in range(len(config.START_DATES)):
        start_date = config.START_DATES[year_idx]
        end_date = config.END_DATES[year_idx]

        for date in daterange(datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")):
            folder = os.path.join(os.path.join(
                data_path, date.strftime("%Y_%m_%d")))
            request_bands = SentinelHubRequest(
                evalscript=evalscript_true_color,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_source=DataSource.SENTINEL2_L1C,
                        time_interval=(date, date + timedelta(days=config.ORBITAL_PERIOD)),
                        maxcc=0.0  # maximum cloud coverage of 0%

                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.PNG)
                ],
                bbox=bbox,
                size=size,
                data_folder=folder,
                config=authorize()
            )

            # request the bands and save as numpy file
            try:
                bands = request_bands.get_data()
                bands_data = bands[0].astype(int)
                height, width, channels = bands_data.shape
                if ((np.count_nonzero(bands_data) / (height * width * channels)) < 0.5):
                    continue

                request_bands.save_data()
                np.save(os.path.join(folder, "bands.npy"), bands_data)

            except Exception as e:
                print(f'Could not fetch an image for date {date}.', e)


if __name__ == '__main__':
    pull_images(config.COORDS_KOENIGSFORST)
