import pandas as pd
import plotly.express as px  # Used for reading the dataset
import streamlit as st
import yfinance as yfin
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, classification_report

pageTitle = "Electric Vehicle Sentiment Analysis & Predictions"
pageIcon = "📈"
pageLayout = "wide"

st.set_page_config(page_title=pageTitle, page_icon=pageIcon, layout=pageLayout)
st.title(pageTitle + " " + pageIcon)

# Importing Tata Motors from YFinance.
tata = yfin.Ticker("TATAMOTORS.NS").history("max")
tata["Date"] = tata.index

final = pd.read_csv("data/FinalDataset.csv", encoding = 'latin1') # "data/car_data.csv"

# https://github.com/mukul-mschauhan/Stock-Sentiment-Analysis/blob/main/Dashboard2.py#:~:text=data-,FinalDataset,-.csv
#------------------------------------------------------------------------

# ----- DROP DOWN VALUES FOR COMPANY TO SELECT FOR ANALYSIS ------
company = ["Tata Motors"]


# ----- HIDE STREAMLIT STYLE -----
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)



# ----- INPUT & SAVE THE COMPANY NAME -----
st.header(f"Company Name")

with st.form("entry_form"):
    comp = st.columns(1)
    comp[0].selectbox("Select Company:", company, key="company_key")

    submitted = st.form_submit_button("Submit")

    companyName = None  # Default value

    if submitted:
        companyName = str(st.session_state["company_key"])
        #st.write(f"Company Name: {companyName}")
        st.success(f"**{companyName}** selected for Analysis")

    if companyName is not None:
        st.subheader(companyName)

        openColor = "#17BEBB"
        highColor = "#F3C178"
        lowColor = "#FE5E41"
        closeColor = "#420217"


        boxLen = 100
        boxhigh = 100
        openBoxCss = f"background-color: {openColor}; width: {boxLen}px; height: {boxhigh}px; padding: 50px; color: white; text-align: right; white-space: pre-line; font-weight: bold; border-bottom: 30px solid;"
        highColorCss = f"background-color: {highColor}; width: {boxLen}px; height: {boxhigh}px; padding: 50px; color: white; text-align: right; white-space: pre-line; font-weight: bold; border-bottom: 30px solid;"
        lowColorCss = f"background-color: {lowColor}; width: {boxLen}px; height: {boxhigh}px; padding: 50px; color: white; text-align: right; white-space: pre-line; font-weight: bold; border-bottom: 30px solid;"
        closeBoxCss = f"background-color: {closeColor}; width: {boxLen}px; height: {boxhigh}px; padding: 50px; color: white; text-align: right; white-space: pre-line; font-weight: bold; border-bottom: 30px solid;"


        col1, col2, col3 = st.columns([1,2,2])
        with col1:
            openValue = int(tata["Open"].iloc[-1])
            highValue = int(tata["High"].iloc[-1])
            lowValue = int(tata["Low"].iloc[-1])
            closeValue = int(tata["Close"].iloc[-1])

            st.markdown(
                f'<div style="{openBoxCss}">OPEN<br>{openValue}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div style="{highColorCss}">HIGH<br>{highValue}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div style="{lowColorCss}">LOW<br>{lowValue}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div style="{closeBoxCss}">CLOSE<br>{closeValue}</div>',
                unsafe_allow_html=True
            )

        with col2:
            st.subheader("Stock Closing Trend Chart")
            if companyName == 'Tata Motors':
                fig = px.line(tata, x="Date", y="Close")
                fig.update_layout(width=550, height=325)
                st.plotly_chart(fig)
            st.subheader("Market Sentiment")
            new = pd.DataFrame({
                "Category": ["Positive","Neutral","Negative"],
                "Value": [28,59,12]
            })
            figure = px.pie(new,names='Category',values='Value')
            figure.update_layout(width=450, height=450)
            st.plotly_chart(figure)

            #    st.markdown("---")

        with col3:
            # Table
            tableCss = """
            table {
              width: 100%;
              border-collapse: collapse;
            }
    
            th, td {
              padding: 8px;
              text-align: left;
              border-bottom: 1px solid #ddd;
            }
    
            th {
              background-color: #f2f2f2;
              font-weight: bold;
            }
            """
            headlines = pd.read_csv("data/FinalHeadlines.csv", encoding='latin1')
            labels = pd.read_csv("data/labels.csv", encoding='latin1')
            ytest = pd.read_csv("data/ytest.csv", encoding='latin1')
            accuracy = accuracy_score(ytest, labels)
            report = classification_report(ytest, labels) 
            st.subheader("Top Headlines")
            #st.markdown(f"<style>{tableCss}</style>", unsafe_allow_html=True)
            st.write(headlines.processed_tweet.value_counts()[:10])
            st.subheader("Model Metrics")
            st.write("Accuracy_Score: ", accuracy)
            # Compute classification report
            report = classification_report(ytest, labels, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            # Format the report table
            df_report["precision"] = df_report["precision"].apply(lambda x: f"{x:.2f}")
            df_report["recall"] = df_report["recall"].apply(lambda x: f"{x:.2f}")
            df_report["f1-score"] = df_report["f1-score"].apply(lambda x: f"{x:.2f}")

            # Display the report table
            st.write("Classification Report:")
            st.dataframe(df_report)

            #st.write("** ")
        
                    
        with st.expander("See Recommendations"):
            col1, col2 = st.columns([1, 3])
            boxLenR = 120
            boxhighR = 75
                # with col1:
                #     recomdColor = "#f40000"
                #     recomdBoxCss = f"background-color: {recomdColor}; width: {boxLenR}px; height: {boxhighR}px; padding: 5px; color: white; text-align: center; white-space: pre-line; font-weight: bold; border-bottom: 10px solid;"
                #     # tab 
                #     st.markdown(
                #         f'<div style="{recomdBoxCss}">RECOMMEND<br>{"SELL"}</div>',
                #         unsafe_allow_html=True
                #     )
                # This is for the comment    
            with col2:
                st.subheader("Prescriptive Advice")
                results = f''' 
                * Basis the Intelligence Drawn from the ML Model and the Sentiment Analysis, We find that the Probable Range for the Market for **1 Day** is **554 – 587**.
                * At the same time, the Price Range for the TAMO for **1 Week** is **535 – 609**.
                * And the Price Range for the TAMO for **1 Month** is **491 – 673**.
                
                Note: The Above ranges are calculated at 95% Confidence Level'''
                st.markdown(results)
            
            with col1:
                recomdColor = "#436436" 
                recomdBoxCss = f"background-color: {recomdColor}; width: {boxLenR}px; height: {boxhighR}px; padding: 5px; color: white; text-align: center; white-space: pre-line; font-weight: bold; border-bottom: 10px solid;"
                st.markdown(
                    f'<div style="{recomdBoxCss}">RECOMMEND<br>{"BUY"}</div>',
                    unsafe_allow_html=True)
            with col2:
                results = f" Market is going up therefore recommendation is to **BUY** the stocks as per the expected value"
                st.markdown(results)

        "----"
        with st.expander("Feedback"):
            comment = st.text_area("", placeholder="Enter your opinion about prediction here....")

        "----"
