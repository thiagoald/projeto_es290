#!/usr/bin/env pipenv run

from importlib import reload
import numpy as np
import folium
import pandas as pd
#import seaborn as sns
from random import sample
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from localization import read_data
from localization import get_distance_in_meters
from localization import gen_regressors
from localization import gen_fingerprints
from localization import cell_search
from localization import gen_taf_struct
from localization import search_taf
from localization import get_errors
from localization import show_stats
from localization import show_stats_graphs
from localization import generate_ta_values
from localization import FilterGrid
from localization import show_box_plots
import matplotlib.pyplot as plt
import time
import random
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import datetime

if __name__ == "__main__":
    #-------------------------RF Fingerprint algorithm-------------------------------------------

    #---Grid resolutions to test
    tests = list(reversed([5, 10, 15, 20]))
    #tests = [20, 50]
    #tests = [10,20,30, 40, 50, 100]
    #tests = [10, 20, 50, 100, 200, 500]

    #tests = range(1,21)
    tests_errors = []
    tests_errors_naive = []
    tests_reduced = []
    tests_time = []
    tests_time_std_devs = []
    tests_time_naive = []
    tests_time_naive_std_devs = []
    offline_tests_time = []
    offline_tests_time_std_devs = []
    offline_tests_time_naive = []
    offline_tests_time_naive_std_devs = []

    num_k_folds = 50

    positions, df_points_all = read_data('resources/data/data2.csv')

    bts_positions, df_btss = read_data('resources/data/dados_BTSs.csv')



    #------------------------------------Sets to K FOLD-------------------------------------------------


    shuffled_df = shuffle(df_points_all)


    size_fold = int(len(shuffled_df) / num_k_folds)
    print(size_fold)

    folds = []
    last = 0

    while last < len(shuffled_df):
        folds.append(shuffled_df[last: last + size_fold])
        last = last + size_fold

    for res in tests:
        
        RESOLUTION = res #meters
        errors = []
        errors_naive = []
        set_lengths = []
        times = []
        times_naive = []
        offline_times = []
        offline_times_naive = []

        #---------------------------------------------------------------------------------------------------
        print("SQUARE GRID GENERATOR")
        print("\tResolution: {} m".format(RESOLUTION))
        
        #------------------------Testing N times to get estatical average------------------------------------
        for i in range(0, num_k_folds):
            #-------------------------------Separate Data------------------------------------------
            
            df_points_test = pd.DataFrame()
            df_points = pd.DataFrame()
            
            for j in range(0, num_k_folds):
                if j == i: #this fold is a test fold
                    df_points_test = folds[j]
                else:
                    df_points = df_points.append(folds[j])
                    
            

            #-------------------------Train Regressores-------------------------------------------
            
            fingerprint_cols = ['pathBTS1','pathBTS2','pathBTS3','pathBTS4','pathBTS5','pathBTS6']
            
            targets = [df_points[t_c] for t_c in fingerprint_cols]
        
            position_columns = ['lat', 'lon']
            
            input_features = df_points[position_columns].values
                    
            scaler = preprocessing.MaxAbsScaler().fit(input_features)
            
            normalized_input_features = scaler.transform(input_features)
            
            regs = gen_regressors(normalized_input_features,targets)
            
            
            #------------------------------Delimiting Area of interesting---------------------------------
            
            lats = np.array(list(df_points['lat']))
            lons = np.array(list(df_points['lon']))
            
            max_lat = lats.max()
            min_lat = lats.min()
            max_lon = lons.max()
            min_lon = lons.min()
            
            #--------------------------------Adjust Grid Resolution-------------------------------------
            
            diff_lat = 0.0
            diff_lon = 0.0
            inc = 0.0000001
            ref_lat = min_lat
            ref_lon = min_lon
            
            while(get_distance_in_meters(ref_lat, ref_lon, ref_lat+diff_lat, ref_lon) < RESOLUTION):
                diff_lat += inc
            while(get_distance_in_meters(ref_lat, ref_lon, ref_lat, ref_lon+diff_lon) < RESOLUTION):
                diff_lon += inc
            
            number_of_squares = res
            
            
            #--------------------------------Creating Squared Grid--------------------------------------
            
            
            grid_lats = np.arange(min_lat, max_lat, diff_lat)

            grid_lons = np.arange(min_lon, max_lon, diff_lon)


            lat_centers = [(l1 + l2)/2 for l1, l2 in zip(grid_lats, grid_lats[1:])]
            lon_centers = [(l1 + l2)/2 for l1, l2 in zip(grid_lons, grid_lons[1:])]
            
            #---------------------------------Creating RF Fingerprints----------------------------------------
            
            cells = [] #CDB
            
            samples = []
            for lat in lat_centers:
                for lon in lon_centers:
                    samples.append((lat, lon))
                    
            #----------------------------------------Normalize data--------------------------------------------
            
            normalized_data = scaler.transform(samples)
            
            #--------------------------------------------------------------------------------------------------
            
            offline_start_time = time.time()
            cells = list(zip(samples, gen_fingerprints(normalized_data, regs)))
            
            #-----------------------------------Matching phase using Timing advance------------------------------
            
            btss = gen_taf_struct(df_btss, cells)
            
            offline_end_time = time.time()
            
            offline_spent_time = offline_end_time - offline_start_time
            
            offline_times.append(offline_spent_time)
            pred_positions_taf = []
            reduced_sets = []

            start_time = time.time()
            
            for point_tas, point_fp in zip(list(df_points_test.values[:, 9:]), list(df_points_test.values[:, 3:9])):
                
                pos, reduced_set  = search_taf(point_tas, point_fp, btss)
                #print(len(reduced_set))
                pred_positions_taf.append(pos)
            
                reduced_sets.append(len(reduced_set))
            
            end_time = time.time()
            spent_time = end_time - start_time
            
            #print(spent_time)
            #print(spent_time.seconds)
            times.append(spent_time)
            
            mean_reduced_set_length = np.mean(reduced_sets)
            
            #print(mean_reduced_set_length)
            
            set_lengths.append(mean_reduced_set_length)
            
            errors_taf = get_errors(df_points_test, pred_positions_taf)
            errors.append(np.mean(errors_taf))
            show_stats(errors_taf)
            
            #------------------------------------------Naive-------------------------------------------------------
            pred_positions_naive = []
            
            offline_naive_start_time = time.time()
            cells_ta = list(zip(samples, gen_fingerprints(normalized_data, regs), generate_ta_values(df_btss, samples)))
            
            offline_naive_end_time = time.time()
            offline_naive_spent_time = offline_naive_end_time - offline_naive_start_time
            offline_times_naive.append(offline_naive_spent_time)
            print(cells_ta[0][2])
            start_time_naive  = time.time()
            
            time_convert_values = []
            for point_tas, point_fp in zip(list(df_points_test.values[:, 9:]),list(df_points_test.values[:, 3:9])):
                
                cells_ta_robson = FilterGrid(cells_ta, point_tas)
                time_convert = time.time()
                aprovado = [(x1,x2) for x1, x2, x3  in cells_ta_robson]
                time_convert_end = time.time()
                time_convert_values.append(time_convert_end-time_convert)

                pos, _ = cell_search(point_fp, aprovado)
                pred_positions_naive.append(pos)
            #print("total_time_convert {0}".format(np.sum(time_convert_values)))
            end_time_naive = time.time()
            spent_time_naive = end_time_naive - start_time_naive
            #print(spent_time_naive)
            #print(spent_time_naive.seconds)
            
            errors_predict_naive = get_errors(df_points_test, pred_positions_naive)
            errors_naive.append(np.mean(errors_predict_naive))
            #show_stats(errors_predict_naive)
        
            times_naive.append(spent_time_naive)
            
            #-------------------------------------------------------------------------------------------------
            
        error_res = np.mean(errors)
        error_res_naive = np.mean(errors_naive)
        
        tests_errors.append(error_res)
        tests_errors_naive.append(error_res_naive)
        
        opa_bb = np.mean(set_lengths)
        reduced_rate = (len(cells) - (opa_bb)) / (len(cells))
        
        tests_reduced.append(reduced_rate)
        
        tests_time.append(np.mean(times))
        tests_time_std_devs.append(np.std(times))
        tests_time_naive.append(np.mean(times_naive))
        tests_time_naive_std_devs.append(np.std(times_naive))
        offline_tests_time.append(np.mean(offline_times))
        offline_tests_time_std_devs.append(np.std(offline_times))
        offline_tests_time_naive.append(np.mean(offline_times_naive))
        offline_tests_time_naive_std_devs.append(np.std(offline_times_naive))
        print("Erro médio utilizando timing advance: para a resolução  {0} : {1}".format(RESOLUTION,error_res))
        
        print("Erro médio utilizando abordagem: para a resolução  {0} : {1}".format(RESOLUTION,error_res_naive))
        
        print("Redução do espaço de busca para a resolução  {0} : {1}".format(RESOLUTION, reduced_rate))
        
        print("Número de células por resolução{0} com medio de células reduzidas{1}".format(len(cells),opa_bb))

        print("Desvio padrao do tempo de treinamento: {} s".format(offline_tests_time_std_devs[-1]))
        print("Desvio padrao do tempo de teste: {} s".format(tests_time_std_devs[-1]))

    #-----------------------------Plot Data-------------------------------------------------------------------

    print("Tempo de offline proposto: {0}".format(offline_tests_time))
    print("Desvios de tempo de offline proposto: {0}".format(offline_tests_time_std_devs))
    print("Tempo de offline filtro por TA:{0}".format(offline_tests_time_naive))
    print("Desvios de tempo de offline filtro por TA: {0}".format(offline_tests_time_naive_std_devs))
    print("Tempo de online Proposto;{0}".format(tests_time))
    print("Desvios de tempo de online Proposto: {0}".format(tests_time_std_devs))
    print("Tempo de online filtro por TA:{0}".format(tests_time_naive))
    print("Desvios de tempo de online filtro por TA: {0}".format(tests_time_naive_std_devs))