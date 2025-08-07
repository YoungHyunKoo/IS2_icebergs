import shapely.geometry
import geopandas as gpd
import numpy as np
from scipy import stats
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
        '--tablepath',
        type=str,
        default="C:\\Users\\yoko2261\\OneDrive - UCB-O365\\IS2_iceberg\\Tables",
        help='Directory for restoring data',
    )
    parser.add_argument(
        '--resultpath',
        type=str,
        default="C:\\Users\\yoko2261\\OneDrive - UCB-O365\\IS2_iceberg\\Profiles",
        help='Directory for saving results',
    )
    parser.add_argument(
        '--latmin',
        type=int,
        default=-80,
        help='Minimum latitude value for bounding box',
    )
    parser.add_argument(
        '--latmax',
        type=int,
        default=-59,
        help='Maximum latitude value for bounding box',
    )
    parser.add_argument(
        '--latw',
        type=float,
        default=0.5,
        help='Boundary width of latitude',
    )
    parser.add_argument(
        '--lonw',
        type=float,
        default=2.0,
        help='Boundary width of latitude',
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=2.0,
        help='Resolution for ATL03 sampling',
    )
    
    args = parser.parse_args()

    return args

######## START ###############################################################
args = parse_args()

year = args.year
tablepath = args.tablepath
resultpath = args.resultpath
lat_min = args.latmin
lat_max = args.latmax
resolution = args.resolution

# Antarctic continent
antacric_file = f"data/USNIC_ANTARC_shelf_2022.shp"
ice_shelf = gpd.read_file(antacric_file).to_crs('EPSG:3976') #.loc[1723:1723, :].reset_index(drop = True)

w = [args.latw, args.lonw]

for lat0 in np.arange(lat_min, lat_max, w[0]*2):
    for lon0 in np.arange(-180+w[1], 180+w[1], w[1]*2):

        # point = gpd.points_from_xy([-108], [-80], crs = 'EPSG:4326').to_crs('EPSG:3976')
        point = pd.DataFrame({"lat": [lat0], "lon": [lon0]})
        point1 = gpd.GeoDataFrame(point, geometry=gpd.points_from_xy(point.lon, point.lat), crs="EPSG:4326").to_crs('EPSG:3976')
        intersect = ice_shelf.sjoin(point1)

        t0 = time.time()
        
        if len(intersect) == 0 and os.path.exists(f"{tablepath}\\{year}\\Iceberg_table_{year}_{lat0}_{lon0}.csv") == False:
        # Not in the Antarctic continent & Not in the generated files
            
            center = [lat0, lon0]        
            gdf = read_ATL03_resample(center, w, year, resolution = resolution)
            t1 = time.time() - t0
            ib_data = pd.DataFrame()
            ib_raw = []
            
            if len(gdf) > 0:
                ib_data, ib_raw = find_icebergs(gdf)
                N = len(ib_data)
                t2 = time.time() - t0 - t1

                ib_data.to_csv(f"{tablepath}/{year}/Iceberg_table_{year}_{lat0}_{lon0}.csv")
                with open(f"{resultpath}/{year}/Iceberg_profile_{year}_{lat0}_{lon0}.pkl", "wb") as output:
                    pickle.dump(ib_raw, output)
                del gdf, ib_data, ib_raw
            
                print(f"{year} - Lat: {lat0}, Lon: {lon0}; Icebergs: {N} ({t1:.1f} + {t2:.1f} seconds)")              
            else:
                print(f"{year} - Lat: {lat0}, Lon: {lon0}; No available ATL03 data ({t1:.1f} seconds)")
                
            
                    
# if len(gdf) > 0:
#     # Display Statistics
#     print("Reference Ground Tracks: {}".format(gdf["rgt"].unique()))
#     print("Cycles: {}".format(gdf["cycle"].unique()))
#     print("Received {} elevations".format(len(gdf)))