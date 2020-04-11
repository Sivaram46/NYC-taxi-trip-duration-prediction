#%% # 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#%% # 2
# read data and store in a dataframe
train_data = pd.read_csv('train.zip', compression = 'zip')
test_data = pd.read_csv('test.zip', compression = 'zip')

train_data.head()

#%% # 3
# converting to respective dtypes
train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])
train_data['dropoff_datetime'] = pd.to_datetime(train_data['dropoff_datetime'])

train_data['store_and_fwd_flag'] = 1 * (train_data['store_and_fwd_flag'] == 'Y')
test_data['store_and_fwd_flag'] = 1 * (test_data['store_and_fwd_flag'] == 'Y')

#%% # 4
# data exploration
print('Ids are unique') if train_data['id'].nunique() == len(train_data['id']) else print('Ids not unique')
print('No missing values') if train_data.count().min() == len(train_data['id']) and test_data.count().min() == len(test_data['id']) else print('There are missing values')

#%% # 5
# plotting the geographical data
N = int(len(train_data) / 10) # since the data is too large. Plot only for 1/10th of data

fig, ax = plt.subplots(ncols = 1, nrows = 1,figsize = (12,10))
plt.xlim(-74.1,-73.7) # longitude
plt.ylim(40.6, 40.9) # latitude

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_facecolor('k')

ax.scatter(train_data['pickup_longitude'].values[:N],train_data['pickup_latitude'].values[:N], c = 'y', s = 0.0009, alpha = 1)

#%% # 6
# removig outliers in trip duration
fig, ax = plt.subplots(ncols = 2)

ax[0].set_title('With outliers')
ax[0].boxplot(np.log(train_data.trip_duration + 1))

q = train_data['trip_duration'].quantile([0.01, 0.99])

train_data = train_data[train_data['trip_duration'] > q.iloc[0]]
train_data = train_data[train_data['trip_duration'] < q.iloc[1]]

ax[1].set_title('Without outliers')
ax[1].boxplot(np.log(train_data.trip_duration + 1)) # transform into log scale
plt.show()

#%% # 7
# distance between pickup and dropoff points can be calculated using haversine formula

def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    diff_lon = lon2 - lon1
    diff_lat = lat2 - lat1

    # haversine formula
    a = np.sin(diff_lat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(diff_lon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a)) # returns central angle of earth
    km = 6367 * c # 6367 for radius of earth in km
    return km


train_data.loc[:,'distance'] = haversine_distance(train_data['pickup_longitude'], train_data['pickup_latitude'],
                         train_data['dropoff_longitude'], train_data['dropoff_latitude'])
test_data.loc[:, 'distance'] = haversine_distance(test_data['pickup_longitude'], test_data['pickup_latitude'],
                         test_data['dropoff_longitude'], test_data['dropoff_latitude'])

# finding average distance for each step using distance and trip duration
train_data.loc[:, 'avg speed'] = 1000 * train_data['distance'] / train_data['trip_duration']

#%% # 8
# finding direction of each trip using basic trigonometry
def find_direction(lat1, lon1, lat2, lon2):
    delta_lon = np.radians(lon2 - lon1)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    y = np.sin(delta_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    return np.degrees(np.arctan2(y, x))

train_data.loc[:, 'direction'] = find_direction(train_data['pickup_latitude'], train_data['pickup_longitude'],
                                                train_data['dropoff_latitude'], train_data['dropoff_longitude'])

test_data.loc[:, 'direction'] = find_direction(test_data['pickup_latitude'], test_data['pickup_longitude'],
                                                test_data['dropoff_latitude'], test_data['dropoff_longitude'])

#%% # 9
# trying with k-fold cross validation for linear regression. Here we choose k = 5
feature_df = train_data[['pickup_latitude', 'pickup_longitude', 'passenger_count', 'distance']]
target_df = train_data[['trip_duration']]

regression = linear_model.LinearRegression()
cv = ShuffleSplit(n_splits = 5, test_size = 0.25, random_state = False)

print(cross_val_score(regression, feature_df, target_df, cv = cv))
# score for linear regression is squared coefficient of determination (R^2).
# we can see that the validation score is bad with linear regression

#%% # 10
# try to fit with ridge regression
test_feature_df = test_data[['pickup_latitude', 'pickup_longitude', 'passenger_count', 'distance']]

ridge_reg = linear_model.Ridge(alpha = 0.5)
ridge_reg.fit(feature_df, target_df)

pred = ridge_reg.predict(test_feature_df)
test_feature_df.loc[:, 'trip_duration'] = pred.astype(int)

#%% # 11
# try with k-means clustering
# since #of data is large, we can use minibatch k-means

coordinates = np.vstack((train_data[['pickup_latitude', 'pickup_longitude']],
                        train_data[['dropoff_latitude', 'dropoff_longitude']],
                        test_data[['pickup_latitude', 'pickup_longitude']],
                        test_data[['dropoff_latitude', 'dropoff_longitude']]))

# take some sample from population and cluseterd it
sample_index = np.random.permutation(len(coordinates))[:500000]
kmeans = MiniBatchKMeans(n_clusters = 80, batch_size = 10000).fit(coordinates[sample_index])

cx = [c[0] for c in kmeans.cluster_centers_]
cy = [c[1] for c in kmeans.cluster_centers_]

#%% # 12
# predict with the fitted k-means clustering. Predict the cluster centers

train_data.loc[:, 'pickup_cluster'] = kmeans.predict(train_data[['pickup_latitude', 'pickup_longitude']])
train_data.loc[:, 'dropoff_cluster'] = kmeans.predict(train_data[['dropoff_latitude', 'dropoff_longitude']])
test_data.loc[:, 'pickup_cluster'] = kmeans.predict(test_data[['pickup_latitude', 'pickup_longitude']])
test_data.loc[:, 'dropoff_cluster'] = kmeans.predict(test_data[['dropoff_latitude', 'dropoff_longitude']])

#%% # 13
# visualize the clusters

fig, ax = plt.subplots(ncols=1, nrows=1)

plt.xlim(-74.1,-73.7) # longitude
plt.ylim(40.6, 40.9) # latitude

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# shading clusters
ax.scatter(train_data['pickup_longitude'].values[:N], train_data['pickup_latitude'].values[:N],
           s=0.02, c=train_data['pickup_cluster'].values[:N], alpha=0.2)
# plotting cluster centers'
ax.scatter(cy, cx, color = 'Black', s = 5, alpha = 1) 

plt.show()

#%% # 14
# some area are always in traffic and some others are crowded. So, it'be helpful if we find avg speed and 
# taxi's count at each unique latitude and longitude values

grpby_cols = ['pickup_latitude', 'pickup_longitude']
stats_in_coords = train_data.groupby(grpby_cols)[['avg speed']].mean()
count_in_coords = train_data.groupby(grpby_cols)['id'].count()
stats_in_coords.loc[:, 'count'] = count_in_coords
stats_in_coords.reset_index()

#%% # 15
# use PCA to transform our latitude and longitude data independent ones
# here we're getting two compoenents from PCA say comp1, comp2 one for latitude and so on...

pca = PCA().fit(coordinates) # coordinates having only latitude and longitude which vertically stacked

train_data.loc[:, 'comp0_pickup'] = pca.transform(train_data[['pickup_latitude', 'pickup_longitude']])[:, 0]
train_data.loc[:, 'comp1_pickup'] = pca.transform(train_data[['pickup_latitude', 'pickup_longitude']])[:, 1]
train_data.loc[:, 'comp0_dropoff'] = pca.transform(train_data[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train_data.loc[:, 'comp1_dropoff'] = pca.transform(train_data[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

test_data.loc[:, 'comp0_pickup'] = pca.transform(test_data[['pickup_latitude', 'pickup_longitude']])[:, 0]
test_data.loc[:, 'comp1_pickup'] = pca.transform(test_data[['pickup_latitude', 'pickup_longitude']])[:, 1]
test_data.loc[:, 'comp0_dropoff'] = pca.transform(test_data[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test_data.loc[:, 'comp1_dropoff'] = pca.transform(test_data[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

# also find manhatten distance for the pca components
train_data.loc[:, 'manhatten_dis_pca'] = np.abs(train_data['comp0_dropoff'] - train_data['comp0_pickup']) + np.abs(train_data['comp1_dropoff'] - train_data['comp1_pickup'])
test_data.loc[:, 'manhatten_dis_pca'] = np.abs(test_data['comp0_dropoff'] - test_data['comp0_pickup']) + np.abs(test_data['comp1_dropoff'] - test_data['comp1_pickup'])

#%% # 16
# By using Open Source Routing Machine (OSRM) of Newyork routes, we can find shortest distance between two points.
# OSRM data are stored in different file. WE can read and use it our model

fast_rout1 = pd.read_csv('C:\\Users\\sivaram\\Documents\\Packages\\Predictive package\\fastest_routes_train_part_1.csv',
                         usecols = ['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fast_rout2 = pd.read_csv('C:\\Users\\sivaram\\Documents\\Packages\\Predictive package\\fastest_routes_train_part_2.csv',
                         usecols = ['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv('C:\\Users\\sivaram\\Documents\\Packages\\Predictive package\\fastest_routes_test.csv',
                               usecols = ['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

train_street_info = pd.concat((fast_rout1, fast_rout2))
train_data = train_data.merge(train_street_info, how = 'left', on = 'id')
test_data = test_data.merge(test_street_info, how = 'left', on = 'id')
train_street_info.head()

#%% # 17
# prepare for modelling

feature_names = ['pickup_cluster', 'dropoff_cluster', 'manhatten_dis_pca', 'comp1_pickup',
                 'total_distance', 'pickup_longitude', 'pickup_latitude', 'distance', 'dropoff_latitude',
                 'total_travel_time', 'direction', 'dropoff_longitude', 'comp1_dropoff', 'comp0_dropoff', 'comp0_pickup']

# logistic transform of trip duraion which is the target variable
y = np.log(train_data['trip_duration'].values + 1) 

#%% # 18
# modelling using xgboost...

Xtrain, Xvalid, ytrain, yvalid = train_test_split(train_data[feature_names].values, y, test_size = 0.2)
dtrain = xgb.DMatrix(Xtrain, label = ytrain)
dvalid = xgb.DMatrix(Xvalid, label = yvalid)

#feature_names.remove('avg speed')
dtest = xgb.DMatrix(test_data[feature_names].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

#%% # 19 
# We can change xgboost's parameters so that get minimal RMSE value

xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,
            'subsample': 0.8, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}


#%% # 20
# finding best RMSE value.
model = xgb.train(xgb_pars, dtrain, 60, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=10)

print('Modeling RMSLE %.5f' % model.best_score)

#%% # 21
# Now we can use our model to predict trip duration for test dataset which is our ultimate goal

ytest = model.predict(dtest)

# Since we've log transformed our trip duration for model fitting, it is necessary to inverse transform 
# to get original trip duraion for test_data
test_data['trip_duration'] = np.exp(ytest) - 1
print('We predict the trip duration for test_data with RMSE error of', model.best_score)
