import shapely.geometry
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
# from shapely.geometry import Polygon, LineString, Point
from scipy import stats
from tqdm import tqdm
import pickle
import time

from functions import *
import argparse

def parse_args() -> argparse.Namespace:    
    # General settings
    parser = argparse.ArgumentParser(description='Argument settings')       
    parser.add_argument(
        '--year',
        type=int,
        default=2020,
        help='Target year',
    )
    parser.add_argument(
        '--datapath',
        type=str,
        default="D:\\Landfast",
        help='Directory for restoring data',
    )
    parser.add_argument(
        '--resultpath',
        type=str,
        default="D:\\IS2_iceberg\\Profiles",
        help='Directory for saving results',
    )
    
    args = parser.parse_args()

    return args

######## START ###############################################################
args = parse_args()

year = args.year
datapath = args.datapath
resultpath = args.resultpath

# Antarctic continent
path = f"{datapath}\\USNIC_ANTARC_shelf_2022.shp"
ice_shelf = gpd.read_file(path).to_crs('EPSG:3976') #.loc[1723:1723, :].reset_index(drop = True)

w = [0.5, 1.0]

for lat0 in np.arange(-80, -60, w[0]*2):
    for lon0 in np.arange(-179, 179, w[1]*2):

        # point = gpd.points_from_xy([-108], [-80], crs = 'EPSG:4326').to_crs('EPSG:3976')
        point = pd.DataFrame({"lat": [lat0], "lon": [lon0]})
        point1 = gpd.GeoDataFrame(point, geometry=gpd.points_from_xy(point.lon, point.lat), crs="EPSG:4326").to_crs('EPSG:3976')
        intersect = ice_shelf.sjoin(point1)

        t0 = time.time()
        
        if len(intersect) == 0: # Not in the Antarctic continent
            center = [lat0, lon0]        
            gdf = read_ATL03_resample(center, w, year)
            t1 = time.time() - t0
            ib_data = pd.DataFrame()
            ib_raw = []
            
            if len(gdf) > 0:
                ib_data, ib_raw = find_icebergs(gdf)
                N = len(ib_data)
                t2 = time.time() - t0 - t1
                print(f"{year} - Lat: {lat0}, Lon: {lon0}; Icebergs: {N} ({t1:.1f} + {t2:.1f} seconds)")              
            else:
                print(f"{year} - Lat: {lat0}, Lon: {lon0}; No available ATL03 data ({t1:.1f} seconds)")
                
            ib_data.to_csv(f"{resultpath}\\Iceberg_table_{year}_{lat0}_{lon0}.csv")
            with open(f"{resultpath}\\Iceberg_profile_{year}_{lat0}_{lon0}.pkl", "wb") as output:
                pickle.dump(ib_raw, output)
            del gdf, ib_data, ib_raw
                    
# if len(gdf) > 0:
#     # Display Statistics
#     print("Reference Ground Tracks: {}".format(gdf["rgt"].unique()))
#     print("Cycles: {}".format(gdf["cycle"].unique()))
#     print("Received {} elevations".format(len(gdf)))