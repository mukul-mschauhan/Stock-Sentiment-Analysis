from cProfile import label
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
from yellowbrick.cluster import KElbowVisualizer
import warnings
#from warnings.filterwarnings("ignore")
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
import plotly as pl
import plotly.express as pex
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.cm as cm
from sklearn.metrics.pairwise import euclidean_distances
# import functions from scipy to perform clustering
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split

# Sidebar Buttons
sbar = st.sidebar.radio(label = "Unsupervised ML", 
                        options = ["IRIS", "Other Dataset"])

# Dataset Imports
