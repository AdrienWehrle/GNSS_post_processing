# -*- coding: utf-8 -*-
""" 

@author: Adrien Wehrlé, University of Oslo (UiO)

Contains different developed (SDF,RF,KMF and KM2FA) and used (EWS and GWS) methods for GNSS post-processing.
For each method, the different steps are run below the initialization of all needed functions. 
For more informations, e.g. a detailed description of the methods, please see the report available here:
https://github.com/AdrienWehrle/GNSS_post_processing/blob/master/WEHRLE_UiO_internship_report.pdf

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import matplotlib.cm as cm
from datetime import datetime, timedelta
import time
import pickle
from kneed import KneeLocator 
from collections import Counter
import scipy.stats as stats
from scipy.stats import linregress
from sklearn.cluster import KMeans 
from scipy.spatial.distance import cdist


def SDF_EM(df, sd_threshold=np.nanmean(df.global_sd), fq='24H', sampling_fq='1H',
           w_min=0, w_max=3 * np.nanstd(df.global_sd), w_step=0.0001,
           temporal_resolution=False, representation=False, save=False):
    ''' Determination of the SDF method's optimal threshold (step 2) with the 
    help of the Elbow Method (from w_min to w_max each w_step) to compute 
    daily velocities (see Adrien Wehrlé's internship report).
    If temporal_resolution==True, velocities are computed for each temporal 
    resolutions (24, 12, 8, 6, 3 and 1 hour).
    If save==True, the resulted velocities are saved in a pickle file 
    (if temporal_resolution==True) or a csv file (if temporal_resolution=False).
    If representation==True, results of the Elbow Method and velocities are 
    displayed.
    '''
    
    def SDF(df, sd_threshold=np.nanmean(df.global_sd), fq='24H', sampling_fq='1H',
            w_min=0, w_max=3 * np.nanstd(df.global_sd)):
        ''' The Standard Deviations Filtering (SDF) is a method that post-processes GNSS measurements
        to obtain the resulting velocities. It consists in four steps:
            1) Ambiguities resolution check 
            2) Global standard deviation filter
            3) Distance to the linear trend filter
            4) Velocity determination
        '''
        
        processing_start = time.clock()
        df_length = len(df) 
         
        def amb_filter(df, init=df_length):
            '''
            Deleting all values where ambiguities are not solved (Q!=1)
            '''
            
            initial_len = len(df)
            df = df.iloc[np.where(df.Q == 1)]
            final_len = len(df)
            nb_del_values = initial_len - final_len
            pc_add_values = nb_del_values / init * 100
            print('amb_filter: %d values removed' % nb_del_values, 
                  '(%.5f %% of the raw data)' % pc_add_values)
            
            return df, nb_del_values
        
        def sd_filter(df, init=df_length, sdthr=sd_threshold):
            '''
            Deleting all values that have a global standard deviation (global_sd) 
            above a given threshold (sdthr).
            The global standard deviation is obtained by combining the standard 
            deviations of each positioning component (X, Y and Z) computed 
            by RTKLIB, at each timestep. 
            '''
            # uncomment if not already determined
            # df['global_sd'] = np.sqrt(df.sde ** 2 + df.sdu ** 2 + df.sdn * *2) 
            initial_len = len(df)
            df = df[df.global_sd < sdthr]
            final_len = len(df)
            nb_del_values = initial_len - final_len
            pc_add_values = nb_del_values / init * 100
            print('sd_filter: %d values have been removed' % nb_del_values, 
                  '(%.5f %% of the raw data)' % pc_add_values)
            
            return df, nb_del_values
        
        def pos_filter(df, init=df_length):
            '''
            Deleting values by computing their minimum distance (mindist) to 
            the linear trend of the station track. 
            Values whose zscored distances are above 1 sigma are deleted. 
            '''
        
            def pos_filter_intermediate(df, init=df_length):
                
                def linear_regression(df):
                    lr_results = linregress(df.X, df.Y)
                    y = df.X * lr_results.slope + lr_results.intercept
                    return y, lr_results
                
                linear_regress, lr_results = linear_regression(df)
                residuals = df.Y - linear_regress
                mindist = np.abs(np.sin(1 - lr_results.slope) * residuals)
                return mindist
            
            mindist = pos_filter_intermediate(df)
            zscore = stats.zscore(mindist)
            df["zscore"] = zscore
            initial_len = len(df)
            df = df[df.zscore < 1]
            final_len = len(df)
            nb_del_values = initial_len - final_len
            pc_add_values = nb_del_values / init * 100
            print('pos_filter: %d values removed' % nb_del_values, 
                  '(%.5f %% of the raw data)' % pc_add_values)
            
            return df, nb_del_values
        
        def appropriate_datetime(df, freq='12H', sampling_freq='1H', nb_col=0, 
                                 mode='mean'):
            '''
            Determine the appropriate datetime while resampling a df with a 
            given timestep (freq) and a targeted precision (sampling_freq).
            Resampling functions (e.g. pd.resample()) always fix the windows 
            indexes to the lowest limit (with a frequency of 12 hours, timesteps 
            are initialy 00:00 and 12:00). It is here proposed to replace these 
            indexes by the mean time (mean mode) or the median time (median mode) 
            of the values contained in each window. 
            nb_col: column index of the datetimes 
            '''
            if mode == 'mean':
                df2 = df.resample(sampling_freq).mean()
                df_modified = pd.DataFrame(df2.resample(freq).mean())
            if mode == 'median':
                df2 = df.resample(sampling_freq).median()
                df_modified = pd.DataFrame(df2.resample(freq).median())
            
            values = df2.iloc[:, nb_col].resample(freq).apply(lambda x: 
                                                              np.where(~np.isnan(x))[-1]) 
        
            res = []
            for i in range(0, len(values)):
                interm = np.array(values.iloc[i])
                res.append(interm)  # necessary to turn resampling results into a df
            res = pd.DataFrame(res)
        
            mean_time_h = res.mean(axis=1, skipna=True)
            mean_time_m = (mean_time_h % 1) * 60  # turn the decimal part into minutes
            
            dt = []
            for i in range(0, len(mean_time_h)):
                if mean_time_h.iloc[i] / mean_time_h.iloc[i] == 1:
                    date = datetime.strptime(str(df_modified.index[i]), 
                                             '%Y-%m-%d %H:%M:%S')
                    if sampling_freq[-1] == 'H':
                        date = date.replace(hour=int(mean_time_h.iloc[i])
                                            + df_modified.index.hour[i])
                    if sampling_freq[-1] == 'T':
                        date = date.replace(hour=int(mean_time_h.iloc[i])
                                            + df_modified.index.hour[i])
                        date = date.replace(minute=int(mean_time_m.iloc[i]
                                                       + df_modified.index.minute[i]))
                    dt.append(date)
                else:
                    dt.append(datetime.strptime(str(df_modified.index[i]), 
                                                '%Y-%m-%d %H:%M:%S'))
            dt = pd.to_datetime(dt, format='%Y-%m-%d %H:%M:%S')
            
            df_modified.index = dt
        
            return df_modified
        
        df['timestamp'] = df.index
        interm_results, nbdv1 = amb_filter(df)
        interm_results, nbdv2 = sd_filter(interm_results)
        data_filt, nbdv3 = pos_filter(interm_results)
        
        nbdv_total = nbdv1 + nbdv2 + nbdv3 
        per_nbdv_total = (nbdv_total / df_length) * 100
        print('\n')
        print('In total, %d values removed' % nbdv_total, 
              '(%.3f %% of the raw data)' % per_nbdv_total)
        data_filt_dt = appropriate_datetime(data_filt, freq=fq, 
                                            sampling_freq=sampling_fq)
        
        # Velocity determination
        deltatime = np.diff(data_filt_dt.index)
        deltatime = pd.DataFrame(deltatime)
        deltatime_dec = deltatime.iloc[:, 0] / timedelta(days=1)
        SDF_velocity = pd.DataFrame((np.sqrt(np.diff(data_filt_dt.X) ** 2 
                                             + np.diff(data_filt_dt.Y) ** 2))
                                             / deltatime_dec)
        SDF_velocity.index = data_filt_dt.index[:-1]
        SDF_vertvelocity = pd.DataFrame(np.diff(data_filt_dt.Z) / deltatime_dec)
        SDF_vertvelocity.index = data_filt_dt.index[:-1]
        
        processing_end = time.clock()
        processing_time = processing_end - processing_start
        if processing_time > 60:
            processing_time = processing_time / 60
            print('KMF method processing time: %.3f minutes' % processing_time)
        if processing_time < 60:
            print('KMF method processing time: %.3f seconds' % processing_time)
        
        return SDF_velocity, SDF_vertvelocity
    
    if temporal_resolution:
        
        freqs = np.array([['24H', '1H'], ['12H', '1H'], ['8H', '1H'], 
                          ['6H', '1H'], ['3H', '1H'], ['1H', '1T']])
        # global_sds = np.arange(0.7 * np.mean(df.global_sd), 
        #                        3 * np.mean(df.global_sd), 0.0001) 
        global_sd_range = np.arange(w_min, w_max, w_step) 
        missval = []
        SDF_velocities = []
        colors = cm.rainbow(np.linspace(0, 1, len(freqs)))
        j = 0
        t = 0 
        
        for k in range(0, len(freqs)):
            missval = []
            
            for i in global_sd_range:
                SDF_velocity, SDF_vertvelocity = SDF(df, sd_threshold=i, 
                                                     fq=freqs[k][0], 
                                                     sampling_fq=freqs[k][1])
                missval.append(np.sum(np.isnan(SDF_velocity)) 
                               / len(SDF_velocity) * 100)
                mv = np.float(np.sum(np.isnan(SDF_velocity)) 
                              / len(SDF_velocity) * 100)
                if mv == 100:
                    t += 1
                    if t == 3:
                        print('FINISHED')
                        break
                print('    %d' % j, '/ %d' % len(global_sd_range))
                j += 1
            
            # a rolling linear regression is used to better determine the elbow point
            if np.int(len(global_sd_range) / 10) % 2 == 0:  # window size must be odd 
                regress = scipy.signal.savgol_filter(missval, 
                                                     np.int(len(global_sd_range) / 10) + 1,
                                                     1) 
            else:
                regress = scipy.signal.savgol_filter(missval, np.int(len(global_sd_range) / 10),
                                                     1)
            # improvment needed to automatically find the curve shape and slope
            kn = KneeLocator(global_sd_range[:j + 1], regress, curve='convex', 
                             direction='decreasing') 
            optimal_sdthreshold = kn.knee
            print('KNEE:', optimal_sdthreshold)
            SDF_velocity, SDF_vertvelocity = SDF(df, sd_threshold=optimal_sdthreshold, 
                                                 fq=freqs[k][0], sampling_fq=freqs[k][1])
            SDF_velocities.append(SDF_velocity)
            
            # results visualisation
            if representation:
                plt.figure()
                plt.subplot(211)
                plt.plot(global_sd_range[:j + 1], missval, color='black', label='', 
                         alpha=0.5)
                plt.plot(global_sd_range[:j + 1], regress, color=colors[k])
                plt.xlabel('Global standard deviation threshold', fontsize=14)
                plt.ylabel('Missing values percentage ($\%$)', fontsize=14)
                plt.axvline(optimal_sdthreshold, label='knee point: %.3f' 
                            % optimal_sdthreshold, color=colors[k])
                plt.subplot(212)
                plt.plot(SDF_velocity.index, SDF_velocity.iloc[:, 0],
                         color='black', alpha=0.5)
                plt.xlabel('Time (year-month)', fontsize=16)
                plt.ylabel('Velocity (meters/day)', fontsize=16)
            print('%d' % k, '/ %d' % len(freqs))
            j = 0
            t = 0
            
        results = ({'24H': SDF_velocities[0], '12H': SDF_velocities[1], 
                    '8H': SDF_velocities[2], '6H': SDF_velocities[3],
                    '3H': SDF_velocities[4], '1H': SDF_velocities[5]})
        
        if save:
            f = open("SDF_method.pkl", "wb")
            pickle.dump(results, f)
            f.close()
        
    else:
        
        global_sd_range = np.arange(w_min, w_max, w_step) 
        missval = []
        colors = cm.rainbow(np.linspace(0, 1, len(freqs)))
        j = 0
        t = 0

        for i in global_sd_range:
            SDF_velocity, SDF_vertvelocity = SDF(df, sd_threshold=i, fq='24H',
                                                 sampling_fq='1H')
            missval.append(np.sum(np.isnan(SDF_velocity)) / len(SDF_velocity) * 100)
            mv = np.float(np.sum(np.isnan(SDF_velocity)) / len(SDF_velocity) * 100)
            if mv == 100:
                t += 1
                if t == 3:
                    print('FINISHED')
                    break
            print('%d' % j, '/ %d' % len(global_sd_range))
            j += 1
            
            # a rolling linear regression is used to better determine the elbow point
            if np.int(len(global_sd_range) / 10) % 2 == 0:  # window size must be odd 
                regress = scipy.signal.savgol_filter(missval, np.int(len(global_sd_range) / 10) + 1,
                                                     1) 
            else:
                regress = scipy.signal.savgol_filter(missval, np.int(len(global_sd_range) / 10),
                                                     1)
            kn = KneeLocator(global_sd_range[:j + 1], regress, curve='convex', 
                             direction='decreasing')
            optimal_sdthreshold = kn.knee
            print('KNEE:', optimal_sdthreshold)
            SDF_velocity, SDF_vertvelocity = SDF(df, sd_threshold=optimal_sdthreshold, 
                                                 fq='24H', sampling_fq='1H')
            results = pd.DataFrame([SDF_velocity, SDF_vertvelocity])
            
            # results visualisation
            if representation:
                plt.figure()
                plt.subplot(211)
                plt.plot(global_sd_range[:j + 1], missval, color='black', 
                         label='', alpha=0.5)
                plt.plot(global_sd_range[:j + 1], regress)
                plt.xlabel('Global standard deviation threshold', fontsize=14)
                plt.ylabel('Missing values percentage ($\%$)', fontsize=14)
                plt.axvline(optimal_sdthreshold, label='knee point: %.3f' % optimal_sdthreshold)
                plt.subplot(212)
                plt.plot(SDF_velocity.index, SDF_velocity.iloc[:, 0], color='black', 
                         alpha=0.5)
                plt.xlabel('Time (year-month)', fontsize=16)
                plt.ylabel('Velocity (meters/day)', fontsize=16)

        if save:
            results.to_csv('SDF_method.csv')
        
    return results




def RF_EM(df, ratio_threshold=np.nanmean(df.ratio), w_min=0, 
          w_max=3*np.nanstd(df.ratio), w_step=0.5, temporal_resolution=False,
          representation=False, save=False):
    ''' Determination of the RF method's optimal threshold (step 1) with the 
    help of the Elbow Method (from w_min to w_max each w_step) to compute daily 
    velocities (see Adrien Wehrlé's internship report).
    If temporal_resolution=True, velocities are computed for each temporal 
    resolutions (24, 12, 8, 6, 3 and 1 hour).
    If representation=True, results of the Elbow Method and velocities are displayed.
    If save=True, the resulted velocities are saved in a pickle file (if 
    temporal_resolution=True) or a csv file (if temporal_resolution=False).
    '''

    def RF(df, ratio_threshold=np.nanmean(df.ratio), w_min=0, 
           w_max=3 * np.nanstd(df.ratio), w_step=0.5):
        ''' The Ratio Filtering (RF) is a method that post-processes GNSS measurements
            to obtain the resulting velocities. It consists in three steps:
                1) Ratio variable filter 
                2) Distance to the linear trend filter
                3) Velocity determination
            '''
            
        processing_start = time.clock()
        df_length = len(df) 
    
        def ratio_filter(df, ratiothr=ratio_threshold, init=df_length):
            '''
            Deleting all values that have a ratio value (global_sd) above a 
            given threshold (ratiothr).
            The global standard deviation is obtained by combining the standard 
            deviations of each positioning component (X, Y and Z) computed by 
            RTKLIB, at each timestep. 
            '''
            initial_len = len(df)
            df = df[df.ratio > ratiothr]
            final_len = len(df)
            nb_del_values = initial_len - final_len
            pc_add_values = nb_del_values / init * 100
            print('sd_filter: %d values have been removed' % nb_del_values, 
                  '(%.5f %% of the raw data)' % pc_add_values)
            
            return df, nb_del_values
        
        def pos_filter(df, init=df_length):
            '''
            Deleting values by computing their minimum distance (mindist) to the linear trend of the station track. 
            Values whose zscored distances are above 1 sigma are deleted. 
            '''
        
            def pos_filter_intermediate(df, init=df_length):
                    
                def linear_regression(df):
                    lr_results = linregress(df.X, df.Y)
                    y = df.X * lr_results.slope + lr_results.intercept
                    return y, lr_results
                    
                linregress, lr_results = linear_regression(df)
                residuals = df.Y - linregress
                mindist = np.abs(np.sin(1 - lr_results.slope) * residuals)
                return mindist
                
            mindist = pos_filter_intermediate(df)
            zscore = stats.zscore(mindist)
            df["zscore"] = zscore
            initial_len = len(df)
            df = df[df.zscore < 1]
            final_len = len(df)
            nb_del_values = initial_len - final_len
            pc_add_values = nb_del_values / init * 100
            print('pos_filter: %d values removed' % nb_del_values, 
                  '(%.5f %% of the raw data)' % pc_add_values)
                
            return df, nb_del_values
        
        def appropriate_datetime(df, freq='12H', sampling_freq='1H', nb_col=0, 
                                 mode='mean'):
            '''
            Determine the appropriate datetime while resampling a df with 
            a given timestep (freq) and a targeted precision (sampling_freq).
            Resampling functions (e.g. pd.resample()) always fix the windows 
            indexes to the lowest limit (with a frequency of 12 hours, 
            timesteps are initialy 00:00 and 12:00). It is here proposed to
            replace these indexes by the mean time (mean mode) or the median 
            time (median mode) of the values contained in each window. 
            nb_col: column index of the datetimes 
            '''
            if mode == 'mean':
                df2 = df.resample(sampling_freq).mean()
                df_modified = pd.DataFrame(df2.resample(freq).mean())
            if mode == 'median':
                df2 = df.resample(sampling_freq).median()
                df_modified = pd.DataFrame(df2.resample(freq).median())
            
            values = df2.iloc[:, nb_col].resample(freq).apply(lambda x: 
                                                              np.where(~np.isnan(x))[-1])
        
            res = []
            for i in range(0, len(values)):
                interm = np.array(values.iloc[i])
                res.append(interm)  # necessary to turn resampling results into a df
            res = pd.DataFrame(res)
        
            mean_time_h = res.mean(axis=1, skipna=True)
            mean_time_m = (mean_time_h % 1) * 60  # turn the decimal part into minutes
            
            dt = []
            for i in range(0, len(mean_time_h)):
                if mean_time_h.iloc[i] / mean_time_h.iloc[i] == 1:
                    date = datetime.strptime(str(df_modified.index[i]), 
                                             '%Y-%m-%d %H:%M:%S')
                    if sampling_freq[-1] == 'H':
                        date = date.replace(hour=int(mean_time_h.iloc[i]) 
                                            + df_modified.index.hour[i])
                    if sampling_freq[-1] == 'T':
                        date = date.replace(hour=int(mean_time_h.iloc[i]) 
                                            + df_modified.index.hour[i])
                        date = date.replace(minute=int(mean_time_m.iloc[i] 
                                                       + df_modified.index.minute[i]))
                    dt.append(date)
                else:
                    dt.append(datetime.strptime(str(df_modified.index[i]), 
                                                '%Y-%m-%d %H:%M:%S'))
            dt = pd.to_datetime(dt, 
                                format='%Y-%m-%d %H:%M:%S')
            
            df_modified.index = dt
        
            return df_modified
     
        df['timestamp'] = df.index
        interm_results, nbdv1 = ratio_filter(df)
        data_filt, nbdv2 = pos_filter(interm_results)
        
        nbdv_total = nbdv1 + nbdv2
        per_nbdv_total = (nbdv_total / df_length) * 100
        print('\n')
        print('In total, %d values removed' % nbdv_total, 
              '(%.3f %% of the raw data)' % per_nbdv_total)
        data_filt_dt = appropriate_datetime(data_filt, freq=fq, 
                                            sampling_freq=sampling_fq)
        
        # velocity determination
        deltatime = np.diff(data_filt_dt.index)
        deltatime = pd.DataFrame(deltatime)
        deltatime_dec = deltatime.iloc[:, 0] / timedelta(days=1)
        RF_velocity = pd.DataFrame((np.sqrt(np.diff(data_filt_dt.X) ** 2 
                                            + np.diff(data_filt_dt.Y) ** 2)) 
                                            / deltatime_dec)
        RF_velocity.index = data_filt_dt.index[:-1]
        RF_vertvelocity = pd.DataFrame(np.diff(data_filt_dt.Z) / deltatime_dec)
        RF_vertvelocity.index = data_filt_dt.index[:-1]
        
        processing_end = time.clock()
        processing_time = processing_end - processing_start
        if processing_time > 60:
            processing_time = processing_time / 60
            print('KMF method processing time: %.3f minutes' % processing_time)
        if processing_time < 60:
            print('KMF method processing time: %.3f seconds' % processing_time)
              
        return RF_velocity, RF_vertvelocity
    
    if temporal_resolution:
        
        freqs = np.array([['24H', '1H'], ['12H', '1H'], ['8H', '1H'], ['6H', '1H'],
                          ['3H', '1H'], ['1H', '1T']])
        ratios_range = np.arange(w_min, w_max, w_step)
        # ratios=np.arange(0, np.mean(df.ratio) + 3 * np.std(df.ratio), 0.5)
        missval = []
        RF_velocities = []
        colors = cm.rainbow(np.linspace(0, 1, len(freqs)))
        j = 0
        t = 0
        
        for k in range(0, len(freqs)):
            missval = []
            
            for i in ratios_range:
                RF_velocity, RF_vertvelocity = RF(df, ratio_threshold=i, 
                                                  fq=freqs[k][0],
                                                  sampling_fq=freqs[k][1])
                missval.append(np.sum(np.isnan(RF_velocity)) 
                               / len(RF_velocity) * 100)
                mv = np.float(np.sum(np.isnan(RF_velocity)) 
                              / len(RF_velocity) * 100)
                if mv == 100:
                    t += 1
                    if t == 3:
                        print('FINISHED')
                        break
                print('    %d' % j, '/ %d' % len(ratios_range))
                j += 1
            
            # a rolling linear regression is used to better determine the elbow point
            if np.int(len(ratios_range) / 10) % 2 == 0:  # window size must be odd 
                regress = scipy.signal.savgol_filter(missval, np.int(len(ratios_range) / 10) + 1,
                                                     1)
            else:
                regress = scipy.signal.savgol_filter(missval, np.int(len(ratios_range) / 10),
                                                     1)
            kn = KneeLocator(ratios_range[:j + 1], regress, curve='convex', 
                             direction='increasing')
            optimal_rthreshold = kn.knee
            print('KNEE:', optimal_rthreshold)
            RF_velocity, RF_vertvelocity = RF(df, ratio_threshold=optimal_rthreshold, 
                                              fq=freqs[k][0], sampling_fq=freqs[k][1])
            RF_velocities.append(RF_velocity)
            
            # results visualisation
            if representation:
                plt.figure()
                plt.subplot(211)
                plt.plot(ratios_range[:j + 1], missval, color='black', label='', 
                         alpha=0.5)
                plt.plot(ratios_range[:j + 1], regress, color=colors[k])
                plt.xlabel('Ratio threshold', fontsize=14)
                plt.ylabel('Missing values percentage ($\%$)', fontsize=14)
                plt.axvline(optimal_rthreshold, label='knee point: %.3f' 
                            % optimal_rthreshold, color=colors[k])
                plt.subplot(212)
                plt.plot(RF_velocity.index, RF_velocity.iloc[:, 0], color='black', 
                         alpha=0.5)
                plt.xlabel('Time (year-month)', fontsize=16)
                plt.ylabel('Velocity (meters/day)', fontsize=16)
                print('%d' % k, '/ %d' % len(freqs))
                j = 0
                t = 0
            
            results = ({'24H': RF_velocities[0], '12H': RF_velocities[1], 
                        '8H': RF_velocities[2], '6H': RF_velocities[3], 
                        '3H': RF_velocities[4], '1H': RF_velocities[5]})
       
            if save:
                f = open("RF_method.pkl", "wb")
                pickle.dump(results, f)
                f.close()
                
        else:
        
            ratios_range = np.arange(w_min, w_max, w_step) 
            missval = []
            colors = cm.rainbow(np.linspace(0, 1, len(freqs)))
            j = 0
            t = 0
    
            for i in ratios_range:
                RF_velocity, RF_vertvelocity = RF(df, ratio_threshold=i, 
                                                  fq='24H', sampling_fq='1H')
                missval.append(np.sum(np.isnan(RF_velocity)) / len(RF_velocity) * 100)
                mv = np.float(np.sum(np.isnan(RF_velocity)) / len(RF_velocity) * 100)
                if mv == 100:
                    t += 1
                    if t == 3:
                        print('FINISHED')
                        break
                print('%d' % j, '/ %d' % len(ratios_range))
                j += 1
                
                # a rolling linear regression is used to better determine the elbow point
                if np.int(len(ratios_range) / 10) % 2 == 0:  # window size must be odd 
                    regress = scipy.signal.savgol_filter(missval, 
                                                         np.int(len(ratios_range) / 10) + 1,
                                                         1) 
                else:
                    regress = scipy.signal.savgol_filter(missval, 
                                                         np.int(len(ratios_range) / 10),
                                                         1)
                kn = KneeLocator(ratios_range[:j + 1], regress, curve='convex', 
                                 direction='increasing')
                optimal_rthreshold = kn.knee
                print('KNEE:', optimal_rthreshold)
                SDF_velocity, SDF_vertvelocity = RF(df, ratio_threshold=optimal_rthreshold,
                                                    fq='24H', sampling_fq='1H')
                results = pd.DataFrame([SDF_velocity, SDF_vertvelocity])
                
                # results visualisation
                if representation:
                    plt.figure()
                    plt.subplot(211)
                    plt.plot(ratios_range[:j + 1], missval, color='black', 
                             label='', alpha=0.5)
                    plt.plot(ratios_range[:j + 1], regress)
                    plt.xlabel('Global standard deviation threshold', fontsize=14)
                    plt.ylabel('Missing values percentage ($\%$)', fontsize=14)
                    plt.axvline(optimal_rthreshold, label='knee point: %.3f' 
                                % optimal_rthreshold)
                    plt.subplot(212)
                    plt.plot(SDF_velocity.index, SDF_velocity.iloc[:, 0],
                             color='black', alpha=0.5)
                    plt.xlabel('Time (year-month)', fontsize=16)
                    plt.ylabel('Velocity (meters/day)', fontsize=16)
    
            if save:
                results.to_csv('RF_method.csv')
                
    return results


def KMF(df, nbc_min=1, nbc_max=10, nb_cycles=5, standardisation=False, 
        variables_importance=False, representation=False, projection='2D', 
        save=False):
    '''The K-Means filtering (KMF) is a method that post-processes GNSS measurements
        to obtain the resulting velocities. It is based on K-means automatic 
        clustering, a basic machine learning algorithm that divides a dataset 
        into k groups by determining the relations between the chosen variables. 
        It consists in six steps:
        1) Computation of the Z-transformed minimum distances 
        2) Determination of the optimal number of clusters by the Elbow Method
        3) K-Means algorithm run 
        4) Clusters filtering 
        5) Determination of the optimal number of cycles by the Elbow Method
        6) Velocity determination
        The variables are initially fixed to the one used in Adrien Wehrlé's work.
        If standardisation=True, the variables are reduced and normalized. 
        If variables_importance=True, a quantification of the variables "usefulness" is led. 
        If representation=True, a scatter plot of the data with the clusters as colors is 
        represented in 2D (projection='2D') or in 3D (projection='3D').
        If save=True, the resulted velocities are saved in a csv file.
    '''
    
    processing_start = time.clock()
    initial_length = len(df) 
    
    def zscore(df):
        '''
        Computing the minimum distance (mindist) to the linear trend of the 
        station track for each value, then z-score transformed.
        '''
        def pos_filter_intermediate(df):
            def linear_regression(df):
                lr_results = linregress(df.X, df.Y)
                y = df.X * lr_results.slope + lr_results.intercept
                return y, lr_results
            
            linear_regress, lr_results = linear_regression(df)
            residuals = df.Y - linear_regress
            mindist = np.abs(np.sin(lr_results.slope) * residuals)
            return mindist
        
        mindist = pos_filter_intermediate(df)
        zscore = stats.zscore(mindist)
        df['zsc_e'] = zscore
        
        return df
    
    def standardisation(KMF_data):
        ''' 
        Standardising the data: centers and reduces each variable.
        '''
        KMF_data = KMF_data.apply(lambda x: (x - x.mean()) / x.std())
        
        return KMF_data
         
    def elbow_point(data, mini=nbc_min, maxi=nbc_max):
        ''' 
        Determine the optimal number of clusters (optimal_nbclusters) with the 
        help of the Elbow Method. KMF algorithm is run with a range of number 
        of clusters (from nbc_min to nbc_max): when plotting the Sum of Squared 
        Error (SSE) as function of the number of clusters, the elbow of the curve
        is the optimal one.
        '''
        
        if standardisation:
            data = standardisation(data)
            
        distortions = []
        K = range(nbc_min, nbc_max)
        
        for k in K:
            
            kmeanModel = KMeans(n_clusters=k).fit(data)
            kmeanModel.fit(data)
            distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / data.shape[0])
            print('K-means fit N° %d solved' % k)
        kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
        optimal_nbclusters = kn.knee
        
        return optimal_nbclusters
    

    def K_means(data,nbclusters):
        '''
        K-means algorithm
        '''
        if standardisation:
            data=standardisation(data)

        kmeans=KMeans(n_clusters=nbclusters) 
        kmeans.fit(data)
        #Algorithm outputs
        centroids=kmeans.cluster_centers_
        labels=kmeans.labels_
        data['label']=labels
        data['X']=df.X
        data['Y']=df.Y
        
        return data,centroids,labels
    
    
    def variables_importance(data_clustered):
        '''
        Quantification of the "usefulness" of each variables used for clustering using ANOVA.
        "We usually examine the means for each cluster on each dimension using
        ANOVA to assess how distinct our clusters are. Ideally, we would obtain 
        significantly different means for most, if not all dimensions, used 
        in the analysis. The magnitude of the F values performed on each 
        dimension is an indication of how well the respective dimension 
        discriminates between clusters." 
        (BURNS, Robert P. et BURNS, Richard. Business research methods and statistics 
        using SPSS. Sage, 2008.)
        '''

        values=np.array(list(Counter(labels)))
        fvalues=[]
        for i in range(0,len(data_clustered.columns)):
            data=data_clustered.iloc[:,i]
            f,p=stats.f_oneway(*[data[data_clustered.label==v] for v in values]) 
            fvalues.append(f)
        fvalues=np.array(fvalues)
        plt.figure()
        plt.semilogy(np.arange(0,len(data_clustered.columns)),fvalues,alpha=0.2,color='black',LineStyle='--',marker='o')
        plt.xlabel('Variables',fontsize=16)
        plt.ylabel('F-value ($\o$)',fontsize=16)
        plt.title('Variables F-values based on ANOVA for each iteration',fontsize=16)
        plt.xticks(np.arange(0,len(data_clustered.columns)),data_clustered.columns)
        
        return fvalues
        

    def filt_kclusters(data_clustered):
        ''' Choose the right clusters. In the case of Adrien Wehrlé's work, the bigger cluster 
        contains all the values to delete (determined after a manual work on the K-means results).
        '''
        
        counter_interm=Counter(data_clustered.label)
        nb_counts=[]
        values=Counter(data_clustered.label)
        values=np.array(list(values))
        for i in range(0,len(counter_interm)):
            nb_counts.append(counter_interm[i])
        nb_counts=pd.DataFrame(nb_counts,columns=['nb_counts'])
        values=pd.DataFrame(values,columns=['val'])
        counter=pd.concat([values,nb_counts],axis=1)
        maxi=np.max(counter.nb_counts) 
        rgroup=counter.val[counter.nb_counts==maxi].index
        filtered_data=data_clustered[data_clustered.label!=rgroup[0]]
    
        return filtered_data
    

    def velocity(data_filtered):
        ''' Velocity determination from the clustered and filtered positions
        '''
        deltatime=np.diff(data_filtered.index)
        deltatime=pd.DataFrame(deltatime)
        deltatime_dec=deltatime.iloc[:,0]/timedelta(days=1)
        KMF_velocity=(np.sqrt(np.diff(data_filtered.X)**2+np.diff(data_filtered.Y)**2))/deltatime_dec
        KMF_velocity=pd.DataFrame(KMF_velocity)
        KMF_velocity.index=data_filtered.index[:-1]
        
        return KMF_velocity

    
    def KMF_correction(data_filtered,nbclusters):
        '''
        In Adrien Wehrlé's work, the data remained under-processed with the initial KMF method due to 
        a high number of outliers.
        It is therefore proposed to run the algorithm nb_cycles times to increase the precision of 
        the filtering. Once more, the optimal number of cycles is determined with the help of the 
        Elbow Method.
        '''
        data_filtered_interm,c,l=K_means(data_filtered.drop(['time'], axis=1),nbclusters)
        data_filtered_interm['time']=df.timestamp
        data_filtered=filt_kclusters(data_filtered_interm)
        
        return data_filtered
    
    def appropriate_datetime(df,freq='12H',sampling_freq='1H',nb_col=0,mode='mean'):
            '''
            Determine the appropriate datetime while resampling a df with a given timestep (freq) 
            and a targeted precision (sampling_freq).
            Resampling functions (e.g. pd.resample()) always fix the windows indexes to the lowest limit
            (with a frequency of 12 hours, timesteps are initialy 00:00 and 12:00). It is here proposed to
            replace these indexes by the mean time (mean mode) or the median time (median mode) of the values 
            contained in each window. 
            nb_col: column index of the datetimes 
            '''
            if mode=='mean':
                df2=df.resample(sampling_freq).mean()
                df_modified=pd.DataFrame(df2.resample(freq).mean())
            if mode=='median':
                df2=df.resample(sampling_freq).median()
                df_modified=pd.DataFrame(df2.resample(freq).median())
            
            values=df2.iloc[:,nb_col].resample(freq).apply(lambda x: np.where(~np.isnan(x))[-1]) #why -1? because it works!
        
            res=[]
            for i in range(0,len(values)):
                interm=np.array(values.iloc[i])
                res.append(interm) #necessary to turn resampling results into a df
            res=pd.DataFrame(res)
        
            mean_time_h=res.mean(axis=1,skipna=True)
            mean_time_m=(mean_time_h%1)*60 #turn the decimal part into minutes
            
            dt=[]
            for i in range(0,len(mean_time_h)):
                if mean_time_h.iloc[i]/mean_time_h.iloc[i]==1:
                    date=datetime.strptime(str(df_modified.index[i]),'%Y-%m-%d %H:%M:%S')
                    if sampling_freq[-1]=='H':
                        date = date.replace(hour=int(mean_time_h.iloc[i])+df_modified.index.hour[i])
                    if sampling_freq[-1]=='T':
                        date = date.replace(hour=int(mean_time_h.iloc[i])+df_modified.index.hour[i])
                        date = date.replace(minute=int(mean_time_m.iloc[i]+df_modified.index.minute[i]))
                    dt.append(date)
                else:
                    dt.append(datetime.strptime(str(df_modified.index[i]),'%Y-%m-%d %H:%M:%S'))
            dt=pd.to_datetime(dt,format='%Y-%m-%d %H:%M:%S')
            
            df_modified.index=dt
        
            return df_modified
    
    data=zscore(df)
    print('Zscore: done')
    #create a dataframe with only the chosen variables
    KMF_data=pd.DataFrame({"ratio":data.ratio,"sdn":data.sdn,"sde":data.sde,"sdu":data.sdu,"zsc_e":data.zsc_e,"ns":data.ns})
    optimal_nbclusters=elbow_point(KMF_data)
    print('Elbow point: done')
    data_clustered,centroids,labels=K_means(KMF_data,nbclusters=optimal_nbclusters)
    print('K-Means clustering: done')
    data_clustered['time']=df.timestamp
    if variables_importance:
        f_values=variables_importance(data_clustered)
    else:
        f_values='not determined'
    data_filtered=filt_kclusters(data_clustered)
    print('K-Means filtering: done')
    
    if representation:
        #Plot every points with the appropriate color, each one corresponding to a different cluster
        #A color vector is created to be sure that similar colors are not used several times.
        colors=["green","red","blue","orange","grey","black","yellow","purple","brown","pink"]
        fig = plt.figure()
        if projection=='3D':
            ax= fig.add_subplot(111, projection='3d')
            
        counter_interm=Counter(labels)
        counter=np.array(list(counter_interm))
    
        for i in counter:
            if projection=='2D':
                plt.scatter(data_clustered.X[data_clustered.label==i],data_clustered.Y[data_clustered.label==i],color=colors[i],alpha=0.5,label='cluster: %d' % i)
            elif projection=='3D':
                ax.scatter(data_clustered.X[data_clustered.label==i],data_clustered.Y[data_clustered.label==i],data_clustered.Z[data_clustered.label==i],color=colors[i],alpha=0.1)
        plt.legend()
        plt.title('Kmeans clustering on positions',fontsize=16)
        plt.xlabel('X positionning (meters)',fontsize=16)    
        plt.ylabel('Y positionning (meters)',fontsize=16) 
        
    KMF_velocities=[]
    data_filtered=appropriate_datetime(data_filtered,freq='24H')
    KMF_velocity=velocity(data_filtered)
    if representation:
        plt.figure()
        plt.plot(KMF_velocity.index,KMF_velocity.iloc[:,0],color='black')
        plt.xlabel('Time (year-month)',fontsize=16)
        plt.ylabel('Velocity (meters/day)',fontsize=16)
        plt.grid(alpha=0.5)
    print('1/%d velocity: done' % nb_cycles)
    KMF_velocities.append(KMF_velocity)
    
    if nb_cycles>1:
        for i in range(0,nb_cycles-1):
            data_filtered=KMF_correction(data_filtered,nbclusters=optimal_nbclusters)
            data_filtered=appropriate_datetime(data_filtered,freq='24H')
            KMF_velocity=velocity(data_filtered)
            KMF_velocities.append(KMF_velocity)
            print('1/%d velocity: done' % i)
            
    #Find the optimal number of cycles with the help of the elbow method applied 
    #to the percentage of missing values
    missval=[]
    for i in range(0,nb_cycles):
        missval.append(np.sum(np.isnan(KMF_velocities[i]))/len(KMF_velocities[i])*100)
    
    nb_cycles_range=np.arange(1,nb_cycles+1)
    kn = KneeLocator(nb_cycles_range,missval,curve='convex', direction='decreasing')
    optimal_nbcycles=kn.knee
    KMF_velocity=KMF_velocities[optimal_nbcycles]
    
    processing_end = time.clock()
    processing_time=processing_end-processing_start
    print('\n')
    if processing_time>60:
        processing_time=processing_time/60
        print('KMF method processing time: %.3f minutes' % processing_time)
    if processing_time<60:
        print('KMF method processing time: %.3f seconds' % processing_time)
    
    final_length=len(data_filtered)
    nbdv=initial_length-final_length
    per_nbdv=(nbdv/initial_length)*100
    print('%d values removed' % nbdv,'(%.3f %% of the raw data)' % per_nbdv)
    if save:
        KMF_velocity.to_csv('KMF_method.csv')
    
    return KMF_velocities,f_values
                



def EWS(df,w1=1.5,w2=0.5,t=5,representation=False,save=False):
    ''' The Exponential Weighted Smoothing (EWS) is a method developed by Helbing, J. (2005) that post-processes GNSS measurements 
        (with a timestep t in seconds) to obtain the resulting velocities. It consists in four steps:
        1) Residuals determination
        2) Residuals exponential smoothing (with a window size w1 in days)
        3) Velocity determination
        4) Velocity exponential smoothing (with a window size w2 in days)
        The window sizes are initially fixed to coincide with J. Helbing's work.
        If representation=True, the resulting velocity is plotted.
        If save=True, the resulted velocities are saved in a csv file.
    '''
    processing_start= time.clock()
    
    def residuals_determination(df,wd=4,ts=5,linear='True'):
        '''
        Determining the residuals of each positioning component in regards to its linear trend. 
        '''
        resx=stats.linregress(df.decimalyear,df.X)
        trend_x=df.decimalyear*resx.slope+resx.intercept
        print('X trend: done! (Rvalue=%.4f)' % resx.rvalue)
        resy=stats.linregress(df.decimalyear,df.Y)
        trend_y=df.decimalyear*resy.slope+resy.intercept
        print('Y trend: done! (Rvalue=%.4f)' % resy.rvalue)
        resz=stats.linregress(df.decimalyear,df.Z)
        trend_z=df.decimalyear*resz.slope+resz.intercept
        print('Z trend: done! (Rvalue=%.4f)' % resz.rvalue)
        
        res_x=df.X-trend_x
        res_y=df.Y-trend_y
        res_z=df.Z-trend_z
        res_xyz=pd.DataFrame({'X':res_x,'Y':res_y,'Z':res_z,'trend_x':trend_x,'trend_y':trend_y,'trend_z':trend_z,'Q':df.Q,'sde':df.sde,'sdu':df.sdu,'sdn':df.sdn,'global_sd':df.global_sd})
        res_xyz.index=df.index
        trend_xyz=pd.DataFrame({'X':trend_x,'Y':trend_y,'Z':trend_z})
                                
        return res_xyz,trend_xyz
    
    def EWM(df,width=w1,timestep=t):
        ''' Exponentially weighted rolling mean to smooth each positioning component.
        width: Window duration in days.
        '''
        coords=['X','Y','Z']
        w=np.int((3600*24*width)/timestep)
        for i in coords:
            df[i]=df[i].ewm(span=w).mean()
        return df
    
    
    res_xyz,trend_xyz=residuals_determination(df)
    res_xyz=res_xyz.resample('5S').mean()
    res_xyz_filt=EWM(res_xyz)
    coords=['X','Y','Z']
    for i in coords:
        res_xyz_filt[i]=trend_xyz[i]+res_xyz_filt[i]
        
    EWS_velocity_interm=np.sqrt(np.diff(res_xyz_filt.X)**2+np.diff(res_xyz_filt.Y)**2)*((3600*24)/5)
    EWS_velocity_interm=pd.DataFrame(EWS_velocity_interm)
    EWS_velocity_interm.index=df.resample('5S').mean().index[:-1]
    EWS_velocity=EWS_velocity_interm.ewm(span=np.int((3600*24*w2)/t)).mean()
    EWS_velocity=pd.DataFrame(EWS_velocity)
    EWS_velocity.index=df.resample('5S').mean().index[:-1]
    
    if representation:
        plt.figure()
        plt.plot(EWS_velocity.index,EWS_velocity)
    
    processing_end = time.clock()
    processing_time=processing_end-processing_start
    print('\n')
    if processing_time>60:
        processing_time=processing_time/60
        print('KMF method processing time: %.3f minutes' % processing_time)
    if processing_time<60:
        print('KMF method processing time: %.3f seconds' % processing_time)
    if save:
        EWS_velocity.to_csv('EWS_method.csv')
    return EWS_velocity


def GWS(df,w=60,representation=False,save=False):
    '''The Gaussian Weighted Smoothing (GWS) is a method developed by Sugiyama, S (2015) 
       that post-processes GNSS measurements to obtain the resulting velocities. 
       It consists in two steps:
       1) Raw data gaussian smoothing (with a window size w in minutes)
       2) Velocity determination
       The window sizes and data timesteps are initially fixed to coincide with S. Sugiyama's work.
       If representation=True, the resulting velocity is plotted.
       If save=True, the resulted velocities are saved in a csv file.
    '''
    data=df.resample('15T').mean()
    X_smoothed=data.X.rolling(w/15,win_type='gaussian').mean(std=np.std(data.X))
    Y_smoothed=data.Y.rolling(w/15,win_type='gaussian').mean(std=np.std(data.Y))
    
    GWS_velocity=(np.sqrt(np.diff(X_smoothed)**2+np.diff(Y_smoothed)**2)/5)*4*24
    GWS_velocity=pd.DataFrame(GWS_velocity)
    GWS_velocity.index=data.index[:-1]
    if representation:
        plt.figure()
        plt.plot(GWS_velocity.index,GWS_velocity)
    if save:
        GWS_velocity.to_csv('GWS_method.csv')
    return GWS_velocity



def KM2FA_DR(data,ratio_threshold=2,nbc_min=1,nbc_max=10,nbins=200,standardisation=True,representation=False,save=False):
    ''' KM2FA Daily Run.
    If standardisation=True, the variables are reduced and normalized. 
    If representation=True, features characteristics are displayed in the form of histograms with a nbins number of bins.
    If save=True, features characteristics are saved in a csv file.
    '''
    def KM2FA(data,nbc_min=1,nbc_max=10,standardisation=True,representation=False,save=False):
        ''' The K-Means Filtering for Features analysis (KM2FA) is a post-processing algorithm developed 
        to describe clouds of points shifted off a GNSS station track.
        It consists in three steps:
        1) Ratio variable filter 
        2) K-Means algorithm run
        3) Features characteristics determination
        The variables and ratio threshold are initially fixed to the one used in Adrien Wehrlé's work.
        '''
        def elbow_point(data,mini=nbc_min,maxi=nbc_max):
            ''' Determine the optimal number of clusters (optimal_nbclusters) with the help of 
            the Elbow Method. KMF algorithm is run with a range of number of clusters (from nbc_min to nbc_max):
            when plotting the Sum of Squared Error (SSE) as function of the number of clusters, the elbow of the curve
            is the optimal one.
            '''
            if standardisation:
                data=standardisation(data)
            distortions = []
            K = range(nbc_min,nbc_max)
            for k in K:
                kmeanModel = KMeans(n_clusters=k).fit(data)
                kmeanModel.fit(data)
                distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
                print('K-means fit N° %d solved' % k)
            kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
            optimal_nbclusters=kn.knee
            
            return optimal_nbclusters
    
        
        def K_means(data,nbclusters):
            '''
            K-means algorithm
            '''
            if standardisation:
                data=standardisation(data)
    
            kmeans=KMeans(n_clusters=nbclusters) 
            kmeans.fit(data)
            #Algorithm outputs
            centroids=kmeans.cluster_centers_
            labels=kmeans.labels_
            data['label']=labels
            data['X']=data.X
            data['Y']=data.Y
            
            return data,centroids,labels
        
        
        def features_characteristics(data_clustered,labels):
            '''
            Determination of the features characteristics in terms of duration (features_duration), time of 
            the day (mean_datetime), ratio variable (mean_ratio), number of satellites (mean_ns),
            Q variable (mean_Q) and number of values (nb_values). 
            '''
            counter_interm=Counter(labels)
            counter=np.array(list(counter_interm))
            features_duration=[] 
            mean_datetime=[]
            mean_ratio=[]
            mean_ns=[]
            mean_Q=[]
            nb_values=[]
            
            for i in counter:
                data_interm=data_clustered[data_clustered.label==i]
                dtime=np.array(data_interm.index.hour)+ np.array(data_interm.index.minute)/60 #in decimal hours
                mean_datetime.append(np.nanmean(dtime))
                mean_ratio.append(np.nanmean(data_interm.ratio))
                mean_ns.append(np.nanmean(data_interm.ns))
                mean_Q.append(np.nanmean(data_interm.Q))
                nbval=len(data_interm)
                nb_values.append(nbval)
                feature_duration=(data_interm.index[-1]-data_interm.index[0])/timedelta(days=1)
                features_duration.append(feature_duration)
                
            features_duration=pd.DataFrame(features_duration)
            results=pd.DataFrame({"features_duration":features_duration.iloc[:,0],"mean_datetime":mean_datetime,"mean_ratio":mean_ratio,"mean_ns":mean_ns,"mean_Q":mean_Q,"nb_values":nb_values}) 
        
            return results
        
        if len(data)!=0:
            kn=elbow_point(data)
            print('Elbow point: done')
            data_clustered,centroids,labels=K_means(data,kn)
            print('K-Means clustering: done')
            results=features_characteristics(data_clustered,labels)
            print('Features characteristics: done')
            print('\n')
        else:
            print('Empty day')
            results=np.empty((1,6))
            results.fill(np.nan)
            print('\n')
        
        
        return results
    
    #data['global_sd']=np.sqrt(data.sde**2+data.sdu**2+data.sdn**2) uncomment if not already determined
    features=data[data.ratio<2]
    perc_features=len(features)/len(data)
    print('Percentage of features in raw data: %.3f' % perc_features)
    KM2FA_data=pd.DataFrame({"X":features.X,"Y":features.Y,"time":features.timestamp,"global_sds":features.global_sd})
    features_characteristics=KM2FA_data.resample('1D').apply(KM2FA)
    if representation:
        legends=['Features duration (hours)','Features mean time of the day (hours)','Features mean ratio ($\o$)','Features mean number of satellites ($\o$)','Features mean Q ($\o$)','Features mean number of values ($\o$)']
        plt.figure()
        plt.suptitle('Features characteristics',fontsize=21)
        for i in range(1,7):
            plt.subplot(3,3,i)
            plt.hist(features_characteristics.iloc[:,i-1]*24,bins=nbins,color='orange')
            plt.xlabel(legends[i-1],fontsize=16)
            plt.ylabel('Occurences ($\o$)',fontsize=16)
        plt.subplots_adjust(hspace=0.5)
    if save:
        features_characteristics.to_csv('KM2FA_DR.csv')
    
    return features_characteristics
