import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as pp
import sklearn.cluster as cluster
import sklearn.metrics as skmet

df = pd.read_csv('stats.csv', skiprows=4)
df = df.fillna(0)
print(df)


#%%
indicator_names = ['Population, total',
                   'Agricultural land (sq. km)',
                   'CO2 emissions (metric tons per capita)',
                   'Energy use (kg of oil equivalent per capita)']

df_filtered = df[df['Indicator Name'].isin(indicator_names)]
print(df_filtered)

years_range = [str(year) for year in range(1990, 2010)]
filtered_df = df_filtered[years_range]

df_filtered = df_filtered.reset_index(drop=True)
print(df_filtered)

# Pivot the data to have countries as rows and indicators as columns


# pivot_df = filtered_df.pivot_table(index='Country Name', columns='Indicator Name')

# print(pivot_df)

# # Remove rows with missing values or non-numeric data
# pivot_df.dropna(axis=0, how='any', inplace=True)
# pivot_df = pivot_df.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')

# # Standardize the data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(pivot_df)

# # Apply K-means clustering
# n_clusters = 3  # Number of clusters
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# cluster_labels = kmeans.fit_predict(scaled_data)

# # Add cluster labels to the DataFrame
# pivot_df['Cluster'] = cluster_labels

# # Print the clustering results
# print("Clustering Results:")
# print(pivot_df[['Cluster']])

# plt.figure(figsize=(10, 6))

# # Define colors for each cluster (adjust as needed)
# colors = ['red', 'green', 'blue']

# # Extract data for plotting
# x = pivot_df['Population, total']
# y = pivot_df['CO2 emissions (metric tons per capita)']

# # Plot each point with its cluster color
# for cluster_id in range(n_clusters):
#     cluster_data = pivot_df[pivot_df['Cluster'] == cluster_id]
#     plt.scatter(cluster_data['Population'], cluster_data['CO2 emissions'], c=colors[cluster_id], label=f'Cluster {cluster_id}')

# # Plot cluster centers (optional)
# cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
# plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, c='black', label='Cluster Centers')

# # Add labels and title
# plt.xlabel('Population')
# plt.ylabel('CO2 emissions')
# plt.title('Clustering of Countries based on Population and CO2 emissions')

# # Add legend
# plt.legend()

# # Show plot
# plt.show()




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

df_clust1 = merged_df[['ag mean first 15','pop mean first 15']]
df_clust2 = merged_df[['co2 mean first 15','en mean first 15']]

# apply the scaling
df_norm1 = scaler1.fit_transform(df_clust1)
df_norm2 = scaler2.fit_transform(df_clust2)

print(df_norm1)
print(df_norm2)


# Normalising latest 15 years
scaler3 = pp.RobustScaler()
scaler4 = pp.RobustScaler()

df_clust3 = last_15_df[['ag mean last 15','pop mean last 15']]
df_clust4 = last_15_df[['co2 mean last 15', 'en mean last 15']]

# apply the scaling
df_norm3 = scaler3.fit_transform(df_clust3)
df_norm4 = scaler4.fit_transform(df_clust4)

print(df_norm3)
print(df_norm4)

#%%
features_df = first_15_df[['ag mean first 15','pop mean first 15']]

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(features_df)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(normalized_data)

# Visualize the clustering results
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Agricultural Land')
plt.ylabel('CO2')
plt.title('Clustering Results')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

#%%
# Cluster for first 15 for Agricultural land v Population
kmeans = cluster.KMeans(n_clusters=3, n_init=100)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm1) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

x = merged_df['ag mean first 15']
y = merged_df['pop mean first 15']
plt.figure(figsize=(8.0, 8.0))
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
#X = df_filtered[['Population, total', 'Agricultural land (sq. km)']]

# # Apply K-means clustering
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)

# # Plot the clusters and centroids
# plt.scatter(X['Population, total'], X['Agricultural land (sq. km)'], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
# plt.xlabel('Population, total')
# plt.ylabel('Agricultural land (sq. km)')
# plt.title('K-means Clustering')
# plt.show()
