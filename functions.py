import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np
import pandas as pd
import h5py  
import os
from sliderule import icesat2
from scipy import stats

# FUNCTION TO READ ATL03 FILES ======================================
def getATL03(fname, beam_number, bbox, maxh = 500):
    # 0, 2, 4 = Strong beam; 1, 3, 5 = weak beam
    
    f = h5py.File(fname, 'r')
    
    orient = f['orbit_info']['sc_orient'][:]  # orientation - 0: backward, 1: forward, 2: transition
    
    if len(orient) > 1:
        print('Transitioning, do not use for science!')
        return [[] for i in beamlist]
    elif (orient == 0):
        beams=['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']                
    elif (orient == 1):
        beams=['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
    # (strong, weak, strong, weak, strong, weak)
    # (beam1, beam2, beam3, beam4, beam5, beam6)

    beam = beams[beam_number]    
    
    try:
        # height of each received photon, relative to the WGS-84 ellipsoid
        # (with some, not all corrections applied, see background info above)
        heights=f[beam]['heights']['h_ph'][:]
        # latitude (decimal degrees) of each received photon
        lats=f[beam]['heights']['lat_ph'][:]
        # longitude (decimal degrees) of each received photon
        lons=f[beam]['heights']['lon_ph'][:]
        # seconds from ATLAS Standard Data Product Epoch. use the epoch parameter to convert to gps time
        deltatime=f[beam]['heights']['delta_time'][:]
    except:
        return pd.DataFrame({})   
    
    if bbox != None:
        if bbox[0] < bbox[2]:
            valid = (lats >= bbox[1]) & (lats <= bbox[3]) & (lons >= bbox[0]) & (lons <= bbox[2])
        else:
            valid = (lats >= bbox[1]) & (lats <= bbox[3]) & ((lons >= bbox[0]) | (lons <= bbox[2]))
    
    if len(heights[valid]) == 0:
        return pd.DataFrame({})    
    else:
        # Surface types for signal classification confidence
        # 0=Land; 1=Ocean; 2=SeaIce; 3=LandIce; 4=InlandWater    
        conf=np.max(f[beam]['heights']['signal_conf_ph'][:, 1:4], axis = 1)
        # confidence level associated with each photon event
        # -2: TEP
        # -1: Events not associated with a specific surface type
        #  0: noise
        #  1: buffer but algorithm classifies as background
        #  2: low
        #  3: medium
        #  4: high
        
        # number of ATL03 20m segments
        n_seg, = f[beam]['geolocation']['segment_id'].shape
        # first photon in the segment (convert to 0-based indexing)
        Segment_Index_begin = f[beam]['geolocation']['ph_index_beg'][:] - 1
        # number of photon events in the segment
        Segment_PE_count = f[beam]['geolocation']['segment_ph_cnt'][:]
        # along-track distance for each ATL03 segment
        Segment_Distance = f[beam]['geolocation']['segment_dist_x'][:]
        # along-track distance (x) for photon events
        x_atc = np.array(f[beam]['heights']['dist_ph_along'][:])
        # cross-track distance (y) for photon events

        # Remove the uneffective reference photons (no geo-correction parameters)
        mask_ind = (Segment_Index_begin >= 0)
        Segment_Index_begin = Segment_Index_begin[mask_ind]
        Segment_PE_count = Segment_PE_count[mask_ind]
        n_seg = len(Segment_PE_count)

        # Geographical correction parameters (refer to the ATL03 documents)
        seg_lat = f[beam]['geolocation/reference_photon_lat'][mask_ind]
        
        dac0 = f[beam]['geophys_corr/dac'][mask_ind]
        dac0[dac0 > 3e+38] = np.nan
        geoid0 = f[beam]['geophys_corr/geoid'][mask_ind]
        geoid0[geoid0 > 3e+38] = np.nan
        
        tide_earth = f[beam]['geophys_corr/tide_earth'][mask_ind]
        tide_earth[tide_earth > 3e+38] = np.nan
        tide_load = f[beam]['geophys_corr/tide_load'][mask_ind]
        tide_load[tide_load > 3e+38] = np.nan
        tide_oc = f[beam]['geophys_corr/tide_ocean'][mask_ind]
        tide_oc[tide_oc > 3e+38] = np.nan
        tide_pole = f[beam]['geophys_corr/tide_pole'][mask_ind]
        tide_pole[tide_pole > 3e+38] = np.nan
        tide_oc_pole = f[beam]['geophys_corr/tide_oc_pole'][mask_ind]
        tide_oc_pole[tide_oc_pole > 3e+38] = np.nan
        tide_eq = f[beam]['geophys_corr/tide_equilibrium'][mask_ind]
        tide_eq[tide_eq > 3e+38] = np.nan
        tide0 = tide_earth + tide_load + tide_oc + tide_pole + tide_oc_pole + tide_eq

        # Remove unaffective geo-correction values
        dac0 = dac0[tide0 != np.nan]
        geoid0 = geoid0[tide0 != np.nan]
        tide0 = tide0[tide0 != np.nan]
        seg_lat = seg_lat[tide0 != np.nan]

        # Since the number of reference points are less than the original photons,
        # reference heights of all photons should be interpolated from the existing reference points
        dac = np.interp(lats, seg_lat, dac0)
        geoid = np.interp(lats, seg_lat, geoid0)
        tide = np.interp(lats, seg_lat, tide0)
        
        # Along track distance = x_atc
        for j in range(n_seg):
            # index for 20m segment j
            idx = Segment_Index_begin[j]
            # number of photons in 20m segment
            cnt = Segment_PE_count[j]
            # add segment distance to along-track coordinates
            x_atc[idx:idx+cnt] += Segment_Distance[j]   


        df03=pd.DataFrame({'beam': beam, 'lat':lats, 'lon':lons, 'x':x_atc,
                           'height':heights, 'dac': dac, 'geoid': geoid, 'tide': tide,
                           'deltatime':deltatime, 'conf':conf
                          })
        
        # select only high-confident photons
        df03 = df03[df03['conf'] == 4].reset_index(drop = True) 
        
        if bbox != None:
            df03 = df03[df03['lat'] >= bbox[1]].reset_index(drop = True)
            df03 = df03[df03['lat'] <= bbox[3]].reset_index(drop = True)
            if bbox[0] < bbox[2]:
                df03 = df03[df03['lon'] >= bbox[0]].reset_index(drop = True)
                df03 = df03[df03['lon'] <= bbox[2]].reset_index(drop = True)
            else:
                df03 = df03[(df03['lon'] >= bbox[0]) | (df03['lon'] <= bbox[2])].reset_index(drop = True)
        
        df03 = df03[df03['height'] <= maxh].reset_index(drop = True) 
        
        return df03

# FUNCTION TO SAMPLE ATL03 DATA FOR EVERY SAME DISTANCE ===============================
def sampling_ATL03(df03, sampled_d = 2, q = 0.5):
    # INPUT
    # - df03: dataframe of ATL03 file
    # - sampled_d: sampled distance (default = 2 m)
    
    df03['sample_id'] = df03['x']//sampled_d    
    grouped = df03[(df03['conf'] >= 4)].groupby(['sample_id'], as_index = False) # high confidence
    sampled = grouped.first()[['lat', 'lon', 'x', 'dac', 'geoid', 'tide']]
    sampled['beam'] = df03['beam'][0]
    sampled['N'] = grouped.count()['lat'] 
    
    if len(sampled) > 0:
        data = grouped.mean()
        sampled['height'] = data['height']
        sampled = sampled[sampled['height'] >= np.quantile(sampled['height'], q)].reset_index(drop=True)
        return sampled
    else:
        return pd.DataFrame({})

def detect_iceberg(df_sam, lb = 5, ub = 100):
    # df_sam: sampled ATL03 height profile (dataframe)
    # lb: lower (minimum) bound of the iceberg height (default: 5 m; https://nsidc.org/cryosphere/quickfacts/icebergs.html)
    # ub: upper (maximum) bound of the iceberg height (default: 100 m)
    
    df_sam.loc[:, "iceberg"] = 0
    
    ## 1. height filter
    h_idx = np.where((df_sam['h_cor'] >= lb) & (df_sam['h_cor'] < ub))[0] # iceberg height (height > 10)
    
    if len(h_idx) > 0:
        ## 2. surrounding height filter
        ib_id = 0 # iceberg id

        # Find the consecutive indices
        ib_start = [0]
        ib_end = []
        
        '''
        for i in range(0, len(h_idx)-1):
            if abs(df_sam['x'][h_idx[i+1]] - df_sam['x'][h_idx[i]]) > 100:
                ib_start.append(h_idx[i+1])
                ib_end.append(h_idx[i])
        ib_end.append(h_idx[-1])
        '''
        
        for i in range(0, len(h_idx)-1):
            if abs(df_sam['x'][h_idx[i+1]] - df_sam['x'][h_idx[i]]) > 100:
                ib_start.append(i+1)
                ib_end.append(i)
        ib_end.append(len(h_idx)-1)        

        for i in range(0, len(ib_start)):
            
            id_l = ib_start[i] # id of the left point
            id_r = ib_end[i] # id of the right point
            
            ib_idx = h_idx[id_l:id_r+1]
#             if (df_sam.loc[max(0, h_idx[id_l]-1), "h_cor"] < lb) and (df_sam.loc[min(h_idx[id_r]+1, len(df_sam)-1), "h_cor"] < lb):
            df_sam.loc[ib_idx, "iceberg"] = 1
            df_sam.loc[ib_idx, "ib_id"] = ib_id
            ib_id += 1
    
    icebergs = df_sam[df_sam["iceberg"] == 1].reset_index(drop=True)
    return icebergs

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

        count, value = np.histogram(gdf['h_mean'], bins = 5000, range = (-100, 100))
        value = value[:-1] + (value[1] - value[2])/2
        mode = value[np.argmax(count)]        
        gdf.loc[:, 'fb'] = gdf['h_mean'].values - mode
        
        gdf = gdf.reset_index()
        gdf.loc[:, 'year'] = gdf['time'].dt.year
        gdf.loc[:, 'month'] = gdf['time'].dt.month
        gdf.loc[:, 'day'] = gdf['time'].dt.day

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

    # Calculate iceberg surface slope
    slope = np.array([1, 1, -1, -1]) / 4    
    dy = np.convolve(y, slope, mode='same') / np.convolve(x, slope, mode='same')
    
    # slope = np.array([1, 0, -1]) / 2
    # ddy = (dy2[1:] - dy2[:-1])
    # ddy2 = np.convolve(dy2, slope, mode='same')

    reg1 = stats.linregress(x, y)
    a1 = reg1.slope
    b1 = reg1.intercept
    r1 = reg1.rvalue
    p1 = reg1.pvalue
    
    reg2 = stats.linregress(x, dy)
    r2 = reg2.rvalue
    p2 = reg2.pvalue

    if p2 < 0.01 and r2 < -0.5:
        ib_class = 1 # dome-shape
    elif p1 < 0.01 and abs(a1) >= 20:
        ib_class = 2 # slopy
    elif p1 >= 0.01 and abs(a1) < 20 and np.nanstd(dy) <= 100:
        ib_class = 3 # Tabular
    else:
        ib_class = 4       

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
                                x = gdf_ib['x_atc'].values * 0.001
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

def determine_landfast(df, model, fields, ind_field = 'ind_fb_0.5', threshold = 100):

    df = df.dropna()

    fields_max = {'fb': 4.0, 'seg_len': 50, 'ph_rate': 20, 'sigma': 0.08, 'fb_std': 1.0,
              'cons_fb': 4.0, 'cons_sigma': 0.08, 'cons_fb_std': 1.0, 'cons_ph_rate': 12, 'cons_std': 1.0,
              'beam_num': 6, 'cnt_fb_0.5': 400, 'sic': 100, 'fb_mode': 1.0}

    df_input = df.copy()
    
    for f in fields:
        values = df[f].values/fields_max[f]
        values[values > 1] = 1
        values[values < 0] = 0
        df_input[f] = values

    df['pred'] = model.predict(df_input.loc[:, fields])
    df.loc[df[ind_field] == 0, 'pred'] = 0
    df.loc[df["sic"] <= 90, 'pred'] = 0
    
    pred_obj = df[[ind_field, 'pred']].groupby(by = ind_field).mean()
    cnt_obj = df[[ind_field, 'pred']].groupby(by = ind_field).count()
    len_obj = df[[ind_field, 'seg_len']].groupby(by = ind_field).sum()

    for ind in pred_obj.index:
        value = pred_obj.loc[ind, 'pred']
        cnt = cnt_obj.loc[ind, 'pred']
        length = len_obj.loc[ind, 'seg_len']
        
        if (value > 0.9) and (length > threshold):
            df.loc[df[ind_field] == ind, 'pred2'] = 1
        else:
            df.loc[df[ind_field] == ind, 'pred2'] = 0

    # Normalization  

    return df