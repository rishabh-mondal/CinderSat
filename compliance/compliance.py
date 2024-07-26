import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import itertools
from scipy.spatial import cKDTree
import geopy.distance
import matplotlib.pyplot as plt
from operator import itemgetter
import os
st.write("Current working directory:", os.getcwd())

# Title of the app
st.title("Automatic Compliance Monitoring for Brick Kilns")

# Dropdown for selecting the state
state = st.selectbox("Select State", ["Punjab", "Haryana", "Bihar"])  # Update the list as needed

# "Uttar_pradesh"

# Checkboxes for different compliance criteria
distance_kilns = st.checkbox("Inter-brick kiln distance < 1km")
distance_hospitals = st.checkbox("Distance to Hospitals < 800m")
distance_water_bodies = st.checkbox("Distance to Water bodies < 500m")
fp2 = "compliance/data/India_State_Boundary.shp"

# Read file using gpd.read_file()
data2 = gpd.read_file(fp2)
# Function to calculate the nearest distances to water bodies
def ckdnearest(brick_kilns, rivers, gdfB_cols=['geometry']):
    A = np.vstack([np.array(geom) for geom in brick_kilns[['lon','lat']].values])
    B = [np.array(geom.coords) for geom in rivers.geometry.to_list()]
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, river_point_idx = ckd_tree.query(A, k=1)
    closest_river_point = B[river_point_idx]
    river_origin_idx = itemgetter(*river_point_idx)(B_ix)
    gdf = pd.concat(
        [brick_kilns, rivers.loc[river_origin_idx, gdfB_cols].reset_index(drop=True),
         pd.DataFrame(closest_river_point, columns = ['closest_river_point_long','closest_river_point_lat']),
         pd.Series(dist, name='dist')], axis=1)
    return gdf
# Function to calculate the nearest distances to hospitals

def ckdnearest_hospital(brick_kilns, hospital_df):
    A = np.vstack([np.array(geom) for geom in brick_kilns[['lon','lat']].values])
    B = np.vstack([np.array(geom) for geom in hospital_df[['Longitude','Latitude']].values])
    ckd_tree = cKDTree(B)
    dist, hospital_idx = ckd_tree.query(A, k=1)
    closest_hospital_point = B[hospital_idx]
    gdf = pd.concat(
        [brick_kilns,
         pd.DataFrame(closest_hospital_point, columns=['closest_hospital_long', 'closest_hospital_lat']),
         pd.Series(dist, name='dist')], axis=1)
    return gdf
# Function to calculate distances between brick kilns and nearest hospitals
def cal_bk_hosp_dist(path, hospital_df):
    state_bk = pd.read_csv(path)
    bk_hospital_mapping = ckdnearest_hospital(state_bk, hospital_df)
    bk_hospital_mapping['distance_km'] = 0
    for i in range(len(bk_hospital_mapping)):
        bk_hospital_mapping['distance_km'][i] = geopy.distance.distance(
            (bk_hospital_mapping['lat'][i], bk_hospital_mapping['lon'][i]),
            (bk_hospital_mapping['closest_hospital_lat'][i], bk_hospital_mapping['closest_hospital_long'][i])
        ).km
    return bk_hospital_mapping

# Load hospitals data
hospital_df = pd.read_csv('compliance/data/India_Hospital_Data.csv')
hospital_df = hospital_df.rename(columns = {'lon' : 'Longitude', 'lat' : 'Latitude'})

# Function to calculate distances between brick kilns and nearest rivers
def cal_bk_river_dist(path, waterways):
    state_bk = pd.read_csv(path)
    bk_river_mapping = ckdnearest(state_bk, waterways)
    bk_river_mapping['distance'] = 0
    for i in range(len(state_bk)):
        bk_river_mapping['distance'][i] = geopy.distance.distance(
            (bk_river_mapping['lat'][i], bk_river_mapping['lon'][i]),
            (bk_river_mapping['closest_river_point_lat'][i], bk_river_mapping['closest_river_point_long'][i])
        ).km
    return bk_river_mapping

# Calculate inter-brick kiln distances

def ckdnearest_brick_kilns(brick_kilns):
    A = np.vstack([np.array(geom) for geom in brick_kilns[['lon','lat']].values])
    ckd_tree = cKDTree(A)
    dist, idx = ckd_tree.query(A, k=2)  # k=2 because the closest point will be itself
    closest_kiln_point = A[idx[:, 1]]  # idx[:, 1] to get the second closest point
    gdf = pd.concat(
        [brick_kilns,
         pd.DataFrame(closest_kiln_point, columns=['closest_kiln_long', 'closest_kiln_lat']),
         pd.Series(dist[:, 1], name='dist')], axis=1)
    return gdf
# Load waterways shapefile
waterways_path = 'compliance/data/waterways.shp'
waterways = gpd.read_file(waterways_path)

# Load brick kilns data (this should be the path to your brick kilns CSV file)
# brick_kilns_path = '/home/patel_zeel/compass24/exact_latlon/haryana.csv'
brick_kilns_paths = {
    "Punjab": 'compliance/data/punjab.csv',
    "Haryana": 'compliance/data/haryana.csv',
    # "Uttar Pradesh": '/home/patel_zeel/kilns_neurips24/exact_latlon/uttar_pradesh.csv',  
    "Bihar": 'compliance/data/bihar.csv',
}

# Load brick kilns data for the selected state
brick_kilns_path = brick_kilns_paths[state]
brick_kilns = pd.read_csv(brick_kilns_path)

bk_river_mapping = cal_bk_river_dist(brick_kilns_path, waterways)
bk_hospital_mapping = cal_bk_hosp_dist(brick_kilns_path, hospital_df)
bk_kiln_mapping = ckdnearest_brick_kilns(pd.read_csv(brick_kilns_path))


brick_kilns['compliant'] = True
if distance_kilns:
    brick_kilns['compliant'] &= bk_kiln_mapping['dist'] >= 1
if distance_hospitals:
    brick_kilns['compliant'] &= bk_hospital_mapping['distance_km'] >= 0.8
if distance_water_bodies:
    brick_kilns['compliant'] &= bk_river_mapping['distance'] >= 0.5

# Plotting the results
fig, ax = plt.subplots(figsize=(8, 6))
# data2 = gpd.read_file(waterways_path)  # Replace this with the appropriate shapefile for the state map
data2.plot(ax=ax, cmap='Pastel2', edgecolor='black', linewidth=0.1)  # State map
waterways.plot(ax=ax, color='blue', linewidth=0.2)  # Water bodies
# Plot all brick kilns in green
brick_kilns_compliant = brick_kilns[brick_kilns['compliant']]
ax.scatter(brick_kilns_compliant['lon'], brick_kilns_compliant['lat'], color='green', s=10, marker='o', label='Compliant Brick Kilns')

# Plot non-compliant brick kilns in red
brick_kilns_non_compliant = brick_kilns[~brick_kilns['compliant']]
ax.scatter(brick_kilns_non_compliant['lon'], brick_kilns_non_compliant['lat'], color='red', s=10, marker='o', label='Non-compliant Brick Kilns')
if state == 'Bihar':
    ax.text(83, 25.8, 'Uttar\n Pradesh')
    ax.text(85.5, 25.5, 'Bihar')
    ax.text(87.9, 25.3, 'West\n Bengal')
    ax.set_xlim(83,89)
    ax.set_ylim(24.25,27)
    
elif state == 'Haryana':
    ax.text(77.3, 29.5, 'Uttar \nPradesh')
    ax.text(74.5, 28.5, 'Rajasthan')
    ax.text(75.5, 30.5, 'Punjab')
    ax.text(76, 29, 'Haryana')
    ax.text(77, 28.6, 'New Delhi')
    ax.set_xlim(74,78)
    ax.set_ylim(27.5,31)
elif state == 'Punjab':
    ax.text(76, 32, 'Himachal\n Pradesh')
    ax.text(75.5, 31, 'Punjab')
    ax.text(74, 29.6, 'Rajasthan')
    ax.text(76, 29.6, 'Haryana')    
    ax.set_xlim(73.5,77)
    ax.set_ylim(29.5,32.5)



plt.legend(loc='upper left')
ax.set_axis_off()
plt.tight_layout(pad=0)

st.pyplot(fig)

# Display the number of non-compliant kilns
num_non_compliant = len(brick_kilns_non_compliant)
st.write(f"Number of non-compliant brick kilns: {num_non_compliant}")

