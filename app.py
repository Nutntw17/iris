# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:17:07 2025

@author: LAB
"""
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

with open ('kmeans_model.pkl','rb')as f:
    loaded_model = pickle.load(f)
    st.set_page_config(page_title='K-means Clustering App', layout='centered')
    st.title("K-Menas Clustering Visualizer by Phatcharadanai Tangoan 6531501084")
    


    X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)
    #predict
    y_kmeans = loaded_model.predict(X)
    
   # Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red')
plt.title('k-Means Clustering')

# Show the plot in Streamlit
st.pyplot(plt)


    
    

