import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as pp
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err

df = pd.read_csv('stats.csv', skiprows=4)
df = df.fillna(0)
print(df)


#%%
indicator_names = ['Population, total',
                   'Agricultural land (sq. km)',
                   'CO2 emissions (metric tons per capita)',
                   'Energy use (kg of oil equivalent per capita)']

df_filtered = df[df['Indicator Name'].isin(indicator_names)]
df_filtered = df_filtered.reset_index(drop=True)
print(df_filtered)


# Function to transpose
def transpose(df_filtered):
    
    ''' Transpose the filtered dataframe '''
    
    transposed_df = df_filtered.transpose()
    return transposed_df

transposed_data = transpose(df_filtered)
print(transposed_data)

#%%
df_ag = pd.read_csv('df_ag.csv')
print(df_ag)

df_co2 = pd.read_csv('df_co2.csv')
print(df_co2)

df_en = pd.read_csv('df_en.csv')
print(df_en)

df_pop = pd.read_csv('df_pop.csv')
print(df_pop)
#%%
first_15_df = pd.DataFrame()
first_15_df['ag mean first 15'] = df_ag['ag mean first 15']
first_15_df['pop mean first 15'] = df_pop['pop mean first 15']
first_15_df['co2 mean first 15'] = df_co2['co2 mean first 15']
first_15_df['en mean first 15'] = df_en['en mean first 15']
print(first_15_df)

last_15_df = pd.DataFrame()
last_15_df['ag mean last 15'] = df_ag['ag mean last 15']
last_15_df['pop mean last 15'] = df_pop['pop mean last 15']
last_15_df['co2 mean last 15'] = df_co2['co2 mean last 15']
last_15_df['en mean last 15'] = df_en['en mean last 15']
print(last_15_df)

merged_df = pd.concat([first_15_df, last_15_df], axis=1)
print(merged_df)

#%%

pd.set_option('display.max_columns', None)
corr = first_15_df.corr(numeric_only=True)
print(corr.round(4))
plt.figure()
plt.imshow(corr)
plt.colorbar()

tick_positions = list(range(len(first_15_df.columns)))
tick_labels = first_15_df.columns
plt.xticks(tick_positions, tick_labels, rotation=30)
plt.yticks(tick_positions, tick_labels, rotation=0)
plt.show()

corr = last_15_df.corr(numeric_only=True)
print(corr.round(4))
plt.figure()
plt.imshow(corr)
plt.colorbar()

tick_positions = list(range(len(last_15_df.columns)))
tick_labels = last_15_df.columns
plt.xticks(tick_positions, tick_labels, rotation=30)
plt.yticks(tick_positions, tick_labels, rotation=0)
plt.show()

#%%
# Normalising first 15 years
scaler1 = pp.RobustScaler()
scaler2 = pp.RobustScaler()
scaler5 = pp.RobustScaler()

df_clust1 = first_15_df[['ag mean first 15','pop mean first 15']]
df_clust2 = first_15_df[['co2 mean first 15','en mean first 15']]
df_clust5 = first_15_df[['co2 mean first 15','ag mean first 15']]

# apply the scaling
df_norm1 = scaler1.fit_transform(df_clust1)
df_norm2 = scaler2.fit_transform(df_clust2)
df_norm5 = scaler2.fit_transform(df_clust5)

print(df_norm1)
print(df_norm2)
print(df_norm5)

# Normalising latest 15 years
scaler3 = pp.RobustScaler()
scaler4 = pp.RobustScaler()
scaler6 = pp.RobustScaler()

df_clust3 = last_15_df[['ag mean last 15','pop mean last 15']]
df_clust4 = last_15_df[['co2 mean last 15', 'en mean last 15']]
df_clust6 = last_15_df[['co2 mean last 15', 'ag mean last 15']]

# apply the scaling
df_norm3 = scaler3.fit_transform(df_clust3)
df_norm4 = scaler4.fit_transform(df_clust4)
df_norm6 = scaler4.fit_transform(df_clust6)

print(df_norm3)
print(df_norm4)
print(df_norm6)

#%%
# features_df = first_15_df[['ag mean first 15','pop mean first 15']]

# # Normalize the data
scaler = StandardScaler()
# normalized_data = scaler.fit_transform(features_df)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(df_norm1)

# Visualize the clustering results
plt.scatter(df_norm1[:, 0], df_norm1[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Agricultural Land')
plt.ylabel('CO2')
plt.title('Clustering Results')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

#%%
# Cluster for first 15 for Agricultural land v Population
kmeans = cluster.KMeans(n_clusters=1, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm1) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

x = first_15_df['ag mean first 15']
y = first_15_df['pop mean first 15']
plt.figure(figsize=(10, 8))
# plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o")
# show cluster centres
plt.scatter(xkmeans, ykmeans, 40, "k", marker="d")
plt.xlabel('Agricultural Land')
plt.ylabel('Population, total')
plt.title('Cluster for Agricultural land v Population for 1990-2004')
plt.show()

#%%
# Cluster for first 15 years for CO2 emission v Energy use
kmeans2 = cluster.KMeans(n_clusters=3, n_init=100)
# Fit the data, results are stored in the kmeans object
kmeans2.fit(df_norm2) # fit done on x,y pairs
# extract cluster labels
labels2 = kmeans2.labels_
# extract the estimated cluster centres and convert to original scales
cen2 = kmeans2.cluster_centers_
cen2 = scaler2.inverse_transform(cen2)
xkmeans2 = cen2[:, 0]
ykmeans2 = cen2[:, 1]

x2 = first_15_df['co2 mean first 15']
y2 = first_15_df['en mean first 15']
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x2, y2, 10, labels2, marker="o")
# show cluster centres
plt.scatter(xkmeans2, ykmeans2, 40, "k", marker="d")
plt.xlabel('CO2 emissions (metric tons per capita)')
plt.ylabel('Energy use (kg of oil)')
plt.title('Cluster for CO2 emission v Energy use for 1990-2004')
plt.show()

#%%
# Cluster for last 15 years for Agricultural land v Population total
kmeans3 = cluster.KMeans(n_clusters=3, n_init=100)
# Fit the data, results are stored in the kmeans object
kmeans3.fit(df_norm3) # fit done on x,y pairs
# extract cluster labels
labels3 = kmeans3.labels_
# extract the estimated cluster centres and convert to original scales
cen3 = kmeans3.cluster_centers_
cen3 = scaler3.inverse_transform(cen3)
xkmeans3 = cen3[:, 0]
ykmeans3 = cen3[:, 1]

x3 = last_15_df['ag mean last 15']
y3 = last_15_df['pop mean last 15']
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x3, y3, 10, labels3, marker="o")
# show cluster centres
plt.scatter(xkmeans3, ykmeans3, 40, "k", marker="d")
plt.xlabel('Agricultural Land')
plt.ylabel('Population, total')
plt.title('Cluster for Agricultural land v Population for 2005-2020')
plt.show()

#%%
# Cluster for first 15 years for Agricultural land v CO2 emission
kmeans5 = cluster.KMeans(n_clusters=1, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans5.fit(df_norm5) # fit done on x,y pairs
# extract cluster labels
labels5 = kmeans5.labels_
# extract the estimated cluster centres and convert to original scales
cen5 = kmeans5.cluster_centers_
cen5 = scaler5.inverse_transform(cen5)
xkmeans5 = cen5[:, 0]
ykmeans5 = cen5[:, 1]

x5 = first_15_df['co2 mean first 15']
y5 = first_15_df['ag mean first 15']
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x5, y5, 10, labels5, marker="o", label='Data points')
# show cluster centres
plt.scatter(xkmeans5, ykmeans5, 40, "k", marker="d", label='Cluster Centre')
plt.xlabel('Agricultural Land')
plt.ylabel('Population, total')
plt.title('Cluster for Agricultural land v Population for 2005-2020')
plt.legend()
plt.grid(True)
plt.show()
#%%

# Cluster for last 15 years for CO2 emission v Energy use
kmeans4 = cluster.KMeans(n_clusters=3, n_init=100)
# Fit the data, results are stored in the kmeans object
kmeans4.fit(df_norm4) # fit done on x,y pairs
# extract cluster labels
labels4 = kmeans4.labels_
# extract the estimated cluster centres and convert to original scales
cen4 = kmeans4.cluster_centers_
cen4 = scaler4.inverse_transform(cen4)
xkmeans4 = cen4[:, 0]
ykmeans4 = cen4[:, 1]

x4 = last_15_df['co2 mean last 15']
y4 = last_15_df['en mean last 15']
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x4, y4, 10, labels4, marker="o")
# show cluster centres
plt.scatter(xkmeans4, ykmeans4, 40, "k", marker="d")
plt.xlabel('CO2 emissions (metric tons per capita)')
plt.ylabel('Energy use (kg of oil)')
plt.title('Cluster for CO2 emissions v Energy use for 2005-2020')
plt.show()

#%%
def one_silhoutte(xy, n):
     
    """ Calculates silhoutte score for n clusters """
    
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy) # fit done on x,y pairs
    labels = kmeans.labels_
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))
    return score

    # calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm1, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
    

    # calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm2, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
    

    # calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm3, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
    

    # calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm4, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
    
#%%
ag_mean = pd.read_csv('agriculture_mean.csv')
print(ag_mean)

plt.figure()
plt.plot(ag_mean['Year'], ag_mean['Mean'], label='CO2 emissions')
plt.ylabel('Agriculture land (km^s)')
plt.xlabel('Year')
plt.title('Agriculture land over 60 years')
plt.legend()
plt.show()

def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 1961
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f

param, covar = opt.curve_fit(poly, ag_mean['Year'], ag_mean['Mean'])
sigma = np.sqrt(np.diag(covar))
print(sigma)
year = np.arange(1961, 2041)
forecast = poly(year, *param)
sigma = err.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma
ag_mean["fit"] = poly(ag_mean['Year'], *param)
plt.figure()
plt.plot(ag_mean['Year'], ag_mean['Mean'], label='Agriculture land')
plt.plot(year, forecast, label="forecast")
# plot uncertainty range
plt.fill_between(year, low, up, color="lime", alpha=0.5)
plt.xlabel('Year')
plt.ylabel('Agriculture')
plt.title('Error for Agriculture land')
plt.legend()
plt.show()

print(f"a = {param[0]:5.3f} +/- {sigma[0]:5.3f}")
print(f"b = {param[1]:5.3f} +/- {sigma[1]:5.3f}")

#%%







