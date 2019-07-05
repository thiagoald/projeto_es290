import folium
import pandas as pd
import numpy as np
from sklearn import linear_model, neighbors
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
import seaborn as sns

def generate_ta_values(bts_list, positions):
    #print(cell_position[0][0])
    #print(bts_list.values[0])
    
    cells_ta = []
    for cell_position in positions:
        ta_values = []
        for bts in bts_list.values:
            bts_pos = bts[1:3]
            dist = int(get_distance_in_meters(cell_position[0], cell_position[1], bts_pos[0], bts_pos[1]))
            ta_values.append(int(dist / 550))
        cells_ta.append(ta_values)
    return cells_ta

def FilterGrid(cells,test_sample):
    subsets = [[] for x in range(7)]
    for cell in cells:
        #print(cell)
        points = 0
        for i in range(6):
            #print(cell[2])
            if cell[2][i] == test_sample[i]:
                points += 1
                #print("deu match")
                #reduced_cells.append(cell)
        subsets[points].append(cell)
    #print(subsets)

    for subset in reversed(subsets):
        if(len(subset) > 0):
            return subset

def get_errors(original_points,pred_positions):
    errors = []
    for (lat, lon), (pred_lat, pred_lon) in zip(list(original_points[original_points.columns[1:3]].values),
                                                pred_positions):
        error = get_distance_in_meters(lat, lon, pred_lat, pred_lon)
        errors.append(error)
    return errors

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
            knn = neighbors.KNeighborsRegressor()
            rssi_reg = GridSearchCV(knn,
                                    {'n_neighbors':list(range(1, 30)), 'weights' : ['distance']},
                                    cv=10,
                                    scoring='neg_mean_squared_error')
        else:
            rssi_reg = linear_model.LinearRegression(normalize=False)
        rssi_reg.fit(samples, target)
        #rssi_reg = rssi_reg.best_estimator_
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
    closest = sorted(cells,key=lambda x: distance(fingerprint, x[1]))[0]
    return closest
#def cell_search_taf(fingerprint, cells):
#    '''Find position'''
    
#    position = np.mean(cell

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
    # return map_

def gen_taf_struct(df_btss, cells):
    btss = []
    for bts in df_btss.values:
        btss.append([])
        # TODO: Substituir 6 pelo TA máximo
        for i in range(10):
            min_dist = i*550
            max_dist = (i + 1)*550
            cells_list = []
            for cell_pos, cell_fing in cells:
                bts_pos = bts[1:3]
                d = get_distance_in_meters(cell_pos[0], cell_pos[1], bts_pos[0], bts_pos[1])
                if min_dist < d < max_dist:
                    cells_list.append((tuple(cell_pos), cell_fing))
            btss[-1].append(cells_list)
    return btss

def search_taf(point_tas,
               point_fp,
               btss):
    # print('TAs:', point_tas)
    cell_set = set(btss[0][int(point_tas[0])])
    # print('Initial search space: ', len(cell_set))
    # TODO: Choose smallest TA to begin with
    for bts_idx, ta in list(enumerate(point_tas))[1:]:
        bts = btss[bts_idx]
        new_cell_set = cell_set.intersection(set(bts[int(ta)]))
        if len(new_cell_set) > 0:
            cell_set = new_cell_set
            #print('After BTS {}: {}'.format(bts_idx, len(cell_set)))
    #print('Final search space: ', len(cell_set))
    cell_set_list = list(cell_set)
    #print(cell_set_list)
    pos = cell_search(point_fp, list(cell_set))
    #print(list(cell_set))
    return pos[0], list(cell_set)

def show_stats(errors):
    print("Min Error (in meters):{}".format(np.min(errors)))
    print("Max Error (in meters):{}".format(np.max(errors)))
    print("Mean Error (in meters):{}".format(np.mean(errors)))
    print("Std. Deviation (in meters):{}".format(np.std(errors)))

def show_stats_graphs(errors):
    # Individual errors
    plt.plot(range(len(errors)),
            errors,
            color='blue',
            linestyle='dashed',
            marker='o',
            markerfacecolor='red',
            markersize=10)
    plt.title('Errors')
    plt.xlabel('Index')
    plt.ylabel('Error (m)')
    plt.show()

    # Histogram
    plt.title('Histogram of errors')
    plt.ylabel('# of samples')
    plt.xlabel('Error (m)')
    plt.axis([0, 2000, 0, 120])
    plt.hist(errors, 10)
    plt.show()

    # Cumulative
    plt.title('Cumulative error')
    plt.xlabel('Error (m)')
    plt.ylabel('% of samples')
    X = np.linspace(1., max(errors), 100)
    Y = []
    for x in X:
        Y.append(100*len([e for e in errors if e<x])/len(errors))

    plt.plot(X, Y)
    plt.show()

def show_box_plots(errors_list, names):
    all_errors = []
    [all_errors.extend(errors) for errors in errors_list]
    labels = []
    [labels.extend([names[i]]*len(errors_list[i])) for i in range(len(errors_list))]
    df_data = {'Error (m)': all_errors,
               'Approach': labels}
    df = pd.DataFrame(df_data)
    plt.title('Boxplots')
    sns.boxplot(x="Approach", y="Error (m)", data=df,width=0.8)
    plt.show()