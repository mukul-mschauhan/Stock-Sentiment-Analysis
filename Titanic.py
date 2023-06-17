from types import CodeType
# Set page title and favicon.
import pandas as pd
import numpy as np
import streamlit as st
st.set_page_config(layout="wide")
#st.set_page_config(page_title="Traingenerator", page_icon=MAGE_EMOJI_URL)
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as pl
import plotly.express as px # Used for reading the dataset
import plotly.graph_objects as go # Used for plotting the Graphs
import plotly.figure_factory as ff
from plotly.offline import plot, init_notebook_mode, iplot # Use plot function to generate the HTML Output of the plots
from collections import Counter


# Containers
header = st.container()
dataset = st.container()
about_data = st.container()
survived = st.container()
barplot = st.container()
pclass = st.container()
gender = st.container()
age = st.container()
cabin = st.container()
family = st.container()
summary = st.container()

# lets Create a Function Called Barchart...
def barcharts(headertext, df, x, y):
    st.subheader(headertext)
    fig = go.Figure([go.Bar(x = df[x], y = df[y], # turquoise, 
                            marker_color = "salmon")]).update_layout(xaxis_title = x, yaxis_title = y)  
    st.write(fig)  

with header:
    st.title("Data Science: Unveiling Titanic's Data Depths 🛳️")
    st.markdown("Here, we will try to derive meaningful insights from the Titanic Data")
    
def callingdata():
    st.header("Decoding Patterns in Titanic ✔️")
    #st.text("Preview of the Dataset")
    titanic = pd.read_csv("~/Documents/Github_Projects/titanic_train.csv") 
    num_var = list(titanic.select_dtypes(include = np.number).columns)
    cat_var = list(titanic.select_dtypes(exclude=np.number).columns)
    results = f" The Numerical Variables are **{num_var}** & \
                Categorical Variables are **{cat_var}** in the data."
    st.write(titanic.head())
    return(st.markdown(results))


with dataset:
    titanic = pd.read_csv("~/Documents/Github_Projects/titanic_train.csv")
    if st.sidebar.checkbox("Dataset 🍿"):
        callingdata()
        st.markdown("---")
        
# Checking the Dataset
with about_data: # Checking the Shape and Size of the Data
    if st.sidebar.checkbox("Shape & Info 🔰"):
        st.header("Lets Check the Info of the Dataset")
        st.write("The Titanic Dataset has: ", 
                 titanic.shape[0], "Rows & ", titanic.shape[1], "columns")
        st.write(titanic.describe())
        st.markdown("---")
        
# lets analyze the Survived Variable 
with survived:
    if st.sidebar.checkbox("Survived 🚢"):
        st.header("🚢 Lets Analyze Survived Variable")
        st.write(titanic.Survived.value_counts(normalize=True))
        st.markdown("From the Above Table, It is clear that the **Survived** Variable is Categorical the Dataset")
        st.markdown("---")
        
with barplot:
    if st.sidebar.checkbox("Plot of Survived 🧭"):
        st.header("	📶 Visualize the Survived Variable")
        gender_count = titanic.Survived.value_counts(normalize=True).reset_index(name = "Count_Survived")
        #males_ = gender_count.iloc[0][0]
        with st.echo(): # st.echo enables us to show the code...
            barcharts("🔝 Barplot - Survived",gender_count, "index", "Count_Survived")
            results = f" Note: The above plot shows that **61.61% Passengers Survived** and **38.38% Passengers Didn't Survive** in the data.**."
        st.markdown(results)
        st.markdown("---")
        
### Whats Next - Here we would now analyze all the variables

# Lets Analyse PClass - PassengerClass

with pclass:
    if st.sidebar.checkbox("PClass vs Survived 🧑‍🤝‍🧑"):
        st.header("🕰️ Let's Analyze the PClass & Survived Together...")
        tbl = pd.crosstab(titanic.Pclass, titanic.Survived).reset_index().melt(id_vars="Pclass", value_name = "Count")
        fig = px.bar(tbl, x = "Pclass", y = "Count", color = "Survived", barmode = "group")
        st.write(fig)
        results = f" Here, We See that **Class 3 Passengers** have higher Fatalities than the rest of the PClass."
        st.markdown(results)
        st.markdown("---")
        
# Relation of Gender with Survival
with gender:
    if st.sidebar.checkbox("Gender vs Survived 	📊"):
        st.header("🚣‍♀️ Lets check if there is any pattern between Gender & Survived")
        tbl = pd.crosstab(titanic.Sex, titanic.Survived).reset_index().melt(id_vars="Sex", value_name = "Count")
        fig = px.bar(tbl, x = "Sex", y = "Count", color = "Survived", barmode = "group")
        st.write(fig)
        results = f" Here, We See that the Category **Males** have higher Fatalities than the Females."
        st.markdown(results)    
        st.markdown("---")
        
# Relation of the Age with the Survived
with age:
    if st.sidebar.checkbox("Age vs Survived 📆"):
        st.header("📆 Lets check if there is any pattern between Age & Survived")
        tbl = pd.crosstab(titanic.Sex, titanic.Survived).reset_index().melt(id_vars="Sex", value_name = "Count")
        one = titanic.loc[titanic.Survived==1]
        zero = titanic.loc[titanic.Survived==0]
        fig = go.Figure()
        fig.add_trace(go.Box(x = zero["Age"], name='Died', marker_color = 'indianred'))
        fig.add_trace(go.Box(x = one["Age"], name='Survived',marker_color = 'lightseagreen'))
        st.write(fig)        
        st.subheader("📆 Statistical Summary of the Age Variable Basis Survived")
        st.write(titanic.groupby("Survived")["Age"].describe()) 
        st.markdown("---")
       
def fam(x):
    if (x>=5):
        return("Large_Family")
    elif(x>=3):
        return("Small_Family")   
    elif(x==2):
        return("Couples")
    else:
        return("Singles")
    
with family:
    if st.sidebar.checkbox("Family 👨‍👩‍👦"):
        st.header("👨‍👩‍👦 Survival of Family")    
        titanic["Family"] = titanic["SibSp"]+titanic.Parch+1
        titanic["Family_Cat"] = titanic.Family.apply(fam)
        tbl = pd.crosstab(titanic.Family_Cat, titanic.Survived).reset_index().melt(id_vars="Family_Cat", value_name = "Count")
        fig = px.bar(tbl, x = "Family_Cat", y = "Count", color = "Survived", barmode = "group")
        st.write(fig)
        result = f' Here, We See that the **Larger Families** have Lower Chances of Survival.'
        st.markdown(result)
        st.markdown("---")
        
cabinlabels = ['C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6',
       'C23 C25 C27', 'B78', 'D33', 'B30', 'C52', 'B28', 'C83', 'F33',
       'F G73', 'E31', 'A5', 'D10 D12', 'D26', 'C110', 'B58 B60', 'E101',
       'F E69', 'D47', 'B86', 'F2', 'C2', 'E33', 'B19', 'A7', 'C49', 'F4',
       'A32', 'B4', 'B80', 'A31', 'D36', 'D15', 'C93', 'C78', 'D35',
       'C87', 'B77', 'E67', 'B94', 'C125', 'C99', 'C118', 'D7', 'A19',
       'B49', 'D', 'C22 C26', 'C106', 'C65', 'E36', 'C54',
       'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'B18', 'C124', 'C91', 'E40',
       'T', 'C128', 'D37', 'B35', 'E50', 'C82', 'B96 B98', 'E10', 'E44',
       'A34', 'C104', 'C111', 'C92', 'E38', 'D21', 'E12', 'E63', 'A14',
       'B37', 'C30', 'D20', 'B79', 'E25', 'D46', 'B73', 'C95', 'B38',
       'B39', 'B22', 'C86', 'C70', 'A16', 'C101', 'C68', 'A10', 'E68',
       'B41', 'A20', 'D19', 'D50', 'D9', 'A23', 'B50', 'A26', 'D48',
       'E58', 'C126', 'B71', 'B51 B53 B55', 'D49', 'B5', 'B20', 'F G63',
       'C62 C64', 'E24', 'C90', 'C45', 'E8', 'B101', 'D45', 'C46', 'D30',
       'E121', 'D11', 'E77', 'F38', 'B3', 'D6', 'B82 B84', 'D17', 'A36',
       'B102', 'B69', 'E49', 'C47', 'D28', 'E17', 'A24', 'C50', 'B42',
       'C148', 'B45', 'B36', 'A21', 'D34', 'A9', 'C31', 'B61', 'C53',
       'D43', 'C130', 'C132', 'C55 C57', 'C116', 'F', 'A29', 'C6', 'C28',
       'C51', 'C97', 'D22', 'B10', 'E45', 'E52', 'A11', 'B11', 'C80',
       'C89', 'F E46', 'B26', 'F E57', 'A18', 'E60', 'E39 E41',
       'B52 B54 B56', 'C39', 'B24', 'D40', 'D38', 'C105']

def cabinx(x):
    if x in cabinlabels:
        return("Avbl")
    else:
        return("Missing")

with cabin:
    if  st.sidebar.checkbox("Cabin 👨‍👩‍👦"):
        titanic["Cabin_Cat"] = titanic.Cabin.apply(cabinx)
        st.header("📍Cabin Vs Survived")
        tbl = pd.crosstab(titanic.Cabin_Cat , titanic.Survived).reset_index().melt(id_vars="Cabin_Cat", value_name = "Count")
        fig = px.bar(tbl, x = "Cabin_Cat", y = "Count", color = "Survived", barmode = "group")
        st.write(fig)
        result = f' Notice that the People who **dont have Cabin have Lower Chances of Survival.**'
        st.markdown(result)
        st.markdown("---")
        
        
with summary:
    if st.sidebar.checkbox("Inference 📑"):
        st.subheader("Inferences & Next Steps")
        notes = f'''
        **We are able to Mine Intelligence on the Biggest Disaster happened in Maritime Industry and found some amazing insights from the data**
        * We saw that **61.6%** of the Passengers Survived where as **38.3%** Passengers Couldn't.
        * All of this pattern is strongly based on his travel details such as Class of the Passenger, Cabin & Gender etc. 
        * Titanic was an unsinkable ship and this hypothesis proved very costly to the makers. This opened the way of research and coming up with more robust infrastructure.
        * The Passengers from **Class 1 have higher chances of Suvival** where as the passengers from Class 3 have very low chances of Survival.
        * Similarly, the passengers who had the Cabin allotted would be most likely Class 1 Passengers and therefore, the passengers with cabin have higher chances of Survival.
        * **Large Families have lower chances of Survival.**
        * We also noticed that there is not much of a difference when it comes to Survival basis age.
        * **The Next Step would be to deal with cleaning of Data, Dealing with Missing Values & then Building a Predictive Model**'''
        st.write(notes)