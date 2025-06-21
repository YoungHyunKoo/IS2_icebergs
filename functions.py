import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np
import pandas as pd
import os
from sliderule import icesat2
from scipy import stats

def read_ATL03_resample(center, w, year, resolution = 2.0):
    lat_max = center[0] - w[0]
    lat_min = center[0] + w[0]
    lon_max = center[1] - w[1]
    lon_min = center[1] + w[1]
    
    region = [ {"lon":lon_min, "lat":lat_min},
               {"lon":lon_max, "lat":lat_min},
               {"lon":lon_max, "lat":lat_max},
               {"lon":lon_min, "lat":lat_max},
               {"lon":lon_min, "lat":lat_min} ]
    
    parms = {
        "poly": region,
        "srt": 2, # Surface type: sea ice
        "cnf": [3,4], # medium & high confidence
        "ats": 1.0,
        "cnt": 4,
        "len": resolution,
        "res": resolution,
        "yapc": {"score": 0, "knn": 0},
        "t0": f"{int(year)}-01-01",
        "t1": f"{int(year)}-12-31",
    }
    
    # Request ATL06 Data
    gdf = icesat2.atl06p(parms)
    
    if len(gdf) > 0:
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y       
        
        gdf = gdf.reset_index()
        gdf.loc[:, 'year'] = gdf['time'].dt.year
        gdf.loc[:, 'month'] = gdf['time'].dt.month
        gdf.loc[:, 'day'] = gdf['time'].dt.day

        # Sea surface height
        count, value = np.histogram(gdf.loc[(gdf['month'] >= 12) | (gdf['month'] <= 3), 'h_mean'], bins = 5000, range = (-100, 100))
        value = value[:-1] + (value[1] - value[2])/2
        mode = value[np.argmax(count)]        
        gdf.loc[:, 'fb'] = gdf['h_mean'].values - mode

        # Remove some unnessary fields
        gdf = gdf.loc[gdf['fb'].values < 100, :].reset_index(drop = True)

        gdf.pop('geometry')
        gdf.pop('w_surface_window_final')
        gdf.pop('pflags')
        gdf.pop('time')

    return gdf

def consecutive_grouping(df0, field = "fb", xfield = "distance", threshold = 10.0):

    N = len(df0)

    cons_cnt = np.zeros(N)
    cons_ind = np.zeros(N)     
    cons_n_ph = np.zeros(N) 
    cons_sigma = np.zeros(N) 
    cons_std = np.zeros(N)
    cons_fb = np.zeros(N)
    check = (df0[field] >= threshold).values
    
    cnt = 0
    ind = 0
    
    if check[0] == 1:
        ind_st = 0
    
    for i in range(1, N):
        if check[i-1] == 0 and check[i] == 1:
            ind_st = i
            cnt += 1
        elif (check[i-1] == 1 and check[i] == 0) or (i == N-1 and check[i] == 1):
            if i == N-1:
                ind_en = N
            else:
                ind_en = i
                
            if ind_en - ind_st > 5:
                ind_st = ind_st+0
                # ind_en = ind_en+1
                if np.nanmax(abs(df0.loc[ind_st:ind_en, xfield].values)) <= 200:
                    ind += 1
                    cons_cnt[ind_st:ind_en] = cnt
                    cons_ind[ind_st:ind_en] = ind
                    cons_n_ph[ind_st:ind_en] = np.nanmedian(df0.loc[ind_st:ind_en, "n_fit_photons"].values)
                    cons_sigma[ind_st:ind_en] = np.nanmedian(df0.loc[ind_st:ind_en, "h_sigma"].values)
                    cons_std[ind_st:ind_en] = np.nanstd(df0.loc[ind_st:ind_en, "fb"].values)
                    cons_fb[ind_st:ind_en] = np.nanmedian(df0.loc[ind_st:ind_en, "fb"].values)
            cnt = 0
        elif check[i-1] == 1 and check[i] == 1:
            cnt += 1

    df0[f"cnt_{field}_{int(threshold)}"] = cons_cnt
    df0[f"ind_{field}_{int(threshold)}"] = cons_ind
    df0[f"cons_sigma"] = cons_sigma
    df0[f"cons_n_ph"] = cons_n_ph
    df0[f"cons_std"] = cons_std
    df0[f"cons_fb"] = cons_fb
    
    return df0

def classify_icebergs(x, y, N = 5):

    sm = np.ones(N)/N
    x = np.convolve(x, sm, mode='valid')
    y = np.convolve(y, sm, mode='valid')
    slope = np.array([1, 1, -1, -1]) / 4
    dy = np.convolve(y, slope, mode='same') / np.convolve(x, slope, mode='same')
    # dx = np.convolve(x, np.ones(4)/4, mode='valid')

    idx = (abs(dy) < 500)
    x = x[idx]
    y = y[idx]
    dy = dy[idx]

    if len(x) > 3:
        reg1 = stats.linregress(x, y)    
        a1 = reg1.slope
        b1 = reg1.intercept
        r1 = reg1.rvalue
        p1 = reg1.pvalue
        
        reg2 = stats.linregress(x, dy)    
        a2 = reg2.slope
        b2 = reg2.intercept
        r2 = reg2.rvalue
        p2 = reg2.pvalue
    
        if p2 < 0.01 and r2 < -0.5 and (a2*x[0]+b2)*(a2*x[-1]+b2) < 0:
            ib_class = 1 # dome-shape
        elif abs(r1) > 0.5 and p1 < 0.01 and abs(a1) >= 20:
            ib_class = 2 # slopy
        elif ((p1 < 0.01 and abs(a1) < 20) or (p1 >= 0.01)) and np.nanmean(abs(dy)) <= 100:
            ib_class = 3 # Tabular
        else:
            ib_class = 4
    else:
        ib_class = 0
        a1 = 0

    return ib_class, a1

def find_icebergs(gdf):
    
    ib_data = pd.DataFrame()      
    ib_raw = []
    k = 0
    
    for rgt in gdf["rgt"].unique():
        for cycle in gdf["cycle"].unique():
            for gt in gdf["gt"].unique():            
                gdf1 = gdf[(gdf["rgt"] == rgt) & (gdf["cycle"] == cycle) & (gdf["gt"] == gt)].reset_index(drop = True)
    
                if len(gdf1) > 0:
                    distance = np.zeros(len(gdf1)) * np.nan
                    distance[1:] = gdf1.loc[1:len(gdf1), 'x_atc'].values - gdf1.loc[0:len(gdf1)-2, 'x_atc'].values
                    gdf1['distance'] = distance            
                    gdf2 = consecutive_grouping(gdf1)
        
                    for ib_idx in gdf2['ind_fb_10'].unique():
                        if ib_idx > 0:
                            gdf_ib = gdf2[(gdf2['ind_fb_10'] == ib_idx) & (gdf2['h_sigma'] < 1.0)].reset_index(drop = True)                        
                            
                            if len(gdf_ib) > 25:
                                ib_raw.append(gdf_ib)
                                x = (gdf_ib['x_atc'].values - gdf_ib['x_atc'].min()) * 0.001
                                y = gdf_ib['fb'].values
                                ib_class, a = classify_icebergs(x, y)
                                ib_data.loc[k, "year"] = np.nanmean(gdf_ib["year"])
                                ib_data.loc[k, "month"] = np.nanmean(gdf_ib["month"])
                                ib_data.loc[k, "day"] = np.nanmean(gdf_ib["day"])
                                ib_data.loc[k, "lat"] = np.nanmean(gdf_ib["lat"])
                                ib_data.loc[k, "lon"] = np.nanmean(gdf_ib["lon"])
                                ib_data.loc[k, "fb_mean"] = np.nanmean(gdf_ib["fb"])
                                ib_data.loc[k, "fb_std"] = np.nanstd(gdf_ib["fb"])
                                ib_data.loc[k, "width"] = gdf_ib['x_atc'].max() - gdf_ib['x_atc'].min()
                                ib_data.loc[k, "fb_slope"] = a
                                ib_data.loc[k, "ib_class"] = ib_class
                                k+=1
                                
    return ib_data, ib_raw