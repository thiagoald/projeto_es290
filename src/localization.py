import folium
import pandas as pd
import numpy as np
from sklearn import linear_model, neighbors
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians

def get_distance_in_meters(lat1_degrees,
                           lon1_degrees,
                           lat2_degrees,
                           lon2_degrees):
    ''' Return distance (in meters) between two pairs of coordinates '''
    # Approximate radius of earth in km
    R = 6373.0
    
    lat1 = radians(lat1_degrees)
    lon1 = radians(lon1_degrees)
    lat2 = radians(lat2_degrees)
    lon2 = radians(lon2_degrees)
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance * 1000

def read_data(file_,
              remove_na=True):
    ''' Return numpy.array with positions (latitude and longitude) and
    pandas.DataFrame with all the data'''
    df = pd.read_csv(file_)
    if remove_na:
        df = df.dropna(axis='index')
    positions = df[['lat', 'lon']]
    return positions.values, df

# TODO: Remove this?
# def plot_line(predicted_lat,
#               predicted_long,
#               targets):
#     ''' Compares predicted data and the correct result'''
#     for p_lat, p_long, (lat, long) in list(zip(predicted_lat,
#                                                predicted_long,
#                                                targets)):
#         print(100*(p_lat - lat)/lat, 100*(p_long - long)/long)
#     plt.plot(predicted_lat,
#              [x for x, _ in targets],
#              marker='x',
#              linestyle='None')
#     plt.show()

def plot_coeffs(lin_reg, columns):
    ''' Plot coefficients of linear regressor in order of importance '''
    coeffs = sorted(zip(columns, lin_reg.coef_),
                    key=lambda x: abs(x[1]),
                    reverse=True)
    plt.xticks(rotation='vertical')
    plt.bar(range(len(coeffs)),
            [abs(c) for _, c in coeffs],
            tick_label=[label for label, _ in coeffs])
    plt.show()

def gen_regressors(samples, targets, params=('knn', 3)):
    ''' Generate a list of regressors given the input and output data
    (samples and targets)'''
    regressors = []
    for target in targets:
        if 'knn' in params:
            rssi_reg = neighbors.KNeighborsRegressor(n_neighbors=params[1])
        else:
            rssi_reg = linear_model.LinearRegression(normalize=True)
        rssi_reg.fit(samples, target)
        regressors.append(rssi_reg)
    return regressors

def gen_fingerprints(samples, regressors):
    ''' Generate a list of fingerprints given the input samples and
    the regressors'''
    rssids = []
    for r in regressors:
        rssids.append(r.predict(samples))
    return zip(*rssids)

def distance(P1, P2):
    ''' Euclidean distance '''
    return sqrt(sum([(c1 - c2)**2 for c1, c2 in zip(P1, P2)]))

def cell_search(fingerprint, cells):
    ''' Find most similar cell, given a fingerprint and a list of cells.
    Each cell is ((lat, lon), fingerprint)'''
    # TODO: Replace sort with another algorithm with a cost inferior to
    # n*log(n)
    closest = sorted(cells,
                     key=lambda x: distance(fingerprint, x[1]))[0]
    return closest

def result_map(positions,
               predicted_positions,
               output_file,
               map_=None,
               only_show_error=False):
    if map_ is None:
        map_ = folium.Map(location=[np.mean([l for l, _ in positions]),
                                    np.mean([l for _, l in positions])],
                          zoom_start=13,
                          tiles='CartoDB dark_matter')
    for i, (pos, pred_pos) in enumerate(zip(positions,
                                            predicted_positions)):
        if not only_show_error:
            marker = folium.CircleMarker(location=pos,
                                         color='green',
                                         weight=2,
                                         radius=1,
                                         fill_color='green',
                                         fill=True)
            marker.add_to(map_)
            marker = folium.CircleMarker(location=pred_pos,
                                         color='yellow',
                                         weight=2,
                                         radius=1,
                                         fill_color='yellow',
                                         fill=True)
            marker.add_to(map_)
        folium.PolyLine([pos, pred_pos],
                        color="red",
                        weight=1,
                        opacity=0.5).add_to(map_)
    map_.save(output_file)
    print('Map saved!')
