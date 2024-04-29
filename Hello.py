# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import streamlit as st
# from streamlit.logger import get_logger

# LOGGER = get_logger(__name__)
# from streamlit.hello.utils import show_code

import streamlit as st 
#st.set_page_config(initial_sidebar_state="collapsed")
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# Base Libraries
import json
import time
import pickle
import numpy as np 
import pandas as pd 
import matplotlib

#Visualization Libraries
import matplotlib.pyplot as plt 
import seaborn as sns 

#Data Standardization Library
from scipy.stats import zscore
from sklearn.utils import resample

#Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#Dimensionality Reduction Library
from sklearn.decomposition import PCA

matplotlib.use('Agg')

from PIL import Image


#Set Title
html_temp = """
    <h1 style="color:black;text-align:center;"> Cancer Tumor Prediction </h1>
    </div>"""

st.markdown(html_temp,unsafe_allow_html=True)

#Function for Bold Heading
def render_subtitle(title_text):
    """Render a subtitle with a tomato background."""
    subtitle_html = (
        "<div style='background-color:tomato;padding:2px;border-radius:5px;'>"
        f"<h2 style='color:white;text-align:center;'>{title_text}</h2>"
        "</div>"
    )
    st.markdown(subtitle_html, unsafe_allow_html=True)
    
def create_heading(title, level):
    """Create a heading in HTML format."""
    heading_style = f"color:tomato;text-align:center;margin:0;padding:1px;border-radius:5px;"
    html_template = f"<h{level} style='{heading_style}'>{title}</h{level}>"
    st.markdown(html_template, unsafe_allow_html=True)
    

def display_welcome_message():
    render_subtitle('Cancer Tumor Prediction App')
    st.write("")

    st.write("This App Predicts Cancer Tumor in Patients Through Machine Learning Algorithms")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()


def home():
    display_welcome_message()
    
    lottie_home = load_lottiefile("home.json")
    st_lottie(
        lottie_home,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        #renderer="svg", # canvas
        height=None,
        width=None,
        key=None,
    )
    
    st.write("")
    
    create_heading("Use the Sidebar for Navigation", 2)


    
def eda():
    #st.subheader("Exploratory Data Analysis")
    render_subtitle("Exploratory Data Analysis")
    st.write("")
    
    df = pd.read_csv("Cancer_Prediction.csv")
    st.dataframe(df.head())
        
    if st.checkbox("Show shape"):
        st.write(df.shape)
    
    if st.checkbox("Show columns"):
        st.write(df.columns)
    
    selected_columns = 0
    if st.checkbox("Select Columns To Show: Only a sample of selected columns will be shown"):
        selected_columns = st.multiselect("Select Columns", df.columns)
        #all_columns = df.columns.tolist()
        df1 = df[selected_columns]
        st.dataframe(df1.sample(5))
    
    if st.checkbox("Show Summary"):
        if selected_columns != 0:
            st.write(df1.describe().T)
        else:
            st.write(df.describe().T)
        
    if st.checkbox("Display Null Values"):
        if selected_columns != 0:
            st.write(df1.isnull().sum())
        else:
            st.write(df.isnull().sum())
        
    if st.checkbox("Display Data Types"):
        if selected_columns != 0:
            st.write(df1.dtypes)
        else:
            st.write(df.dtypes)
        
    if st.checkbox("Display Correlation of Features"):
        if selected_columns != 0:
            st.write(df1.corr(numeric_only=True))
        else:
            st.write(df.corr(numeric_only=True))


def visualization():
    render_subtitle("Visualization")
    st.write("")
    st.write("") 
    
    df = pd.read_csv("Cancer_Prediction.csv")
    st.dataframe(df.head())
    
    st.write("")
    st.write("")
    st.write("")   
        # if st.checkbox("Show Correlation Plot"):
        #     st.write(sns.heatmap(df.corr(), annot=True))
        #     st.pyplot()
    
    selected_columns = 0
    if st.checkbox("Select Multiple Columns to plot"):
        selected_columns = st.multiselect("Select your preferred Columns", df.columns)
        df1 = df[selected_columns]
        st.dataframe(df1.sample(5))
        
    if st.checkbox("Display Heatmap"):
        if selected_columns == 0:
            plt.figure()
            ax = sns.heatmap(df.iloc[:, 0:10].corr(numeric_only=True), annot=True, fmt=".1f", linewidth=.5)
            ax.set(xlabel="", ylabel="")
            #ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(plt)         

        else:
            plt.figure()
            ax = sns.heatmap(df1.corr(numeric_only=True), annot=True, fmt=".1f", linewidth=.5)
            ax.set(xlabel="", ylabel="")
            #ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(plt)
    
    if st.checkbox("Display Pairplot"):
        if selected_columns == 0:
            plt.figure()
            g = sns.PairGrid(df.iloc[:,0:10], diag_sharey=False, corner=True)
            g.map_lower(sns.scatterplot)
            g.map_diag(sns.kdeplot)
            st.pyplot(plt)
        else:
            plt.figure()
            g = sns.PairGrid(df1, diag_sharey=False, corner=True)
            g.map_lower(sns.scatterplot)
            g.map_diag(sns.kdeplot)
            st.pyplot(plt)

    if st.checkbox("Display Pie Chart Distribution of Target Variable"):
        plt.figure()
        pie_column = 'diagnosis' #st.selectbox("Select Column to Display Pie Chart", df.columns)
        pie_chart = df[pie_column].value_counts().plot.pie(autopct="%1.1f%%")
        #st.write(pie_chart)
        st.pyplot(plt)

def pre_processing(df,n_com=1, pca=1):
    st.write("")
    create_heading("Pre-Processing", 3)
    # Removing Unwanted Columns
    for col in ["Unnamed: 32", "id", "diagnosis"]:
        if col in df.columns.to_list():
            df = df.drop([col], axis=1)
            st.write("")
            st.success("Unwanted Columns Removed")
    
    # Standardizing the data
    df_scaled = df.apply(zscore)
    df = pd.DataFrame(df_scaled, columns=df.columns)
    st.success("Data Standardized")
    
    # Eliminating Multicolinearity based on n_features_to_select
    if pca == 1:
        pca = PCA(n_components=n_com)
        pca.fit(df)
        
        #Load into a Pickle File
        model_save_path = "PCA.pkl"
        with open(model_save_path,'wb') as file:
            pickle.dump(pca, file)
        
        with open('PCA.pkl', 'rb') as file:
            R_pca = pickle.load(file)
            
        tpca = R_pca.transform(df)
        st.success(f"PCA Completed, with {n_com} components")
        tpca_df = pd.DataFrame(data = tpca, columns = [f"PC{i}" for i in range(1, n_com + 1)])
        return tpca_df
        
        # tpca = pca.transform(df)
        # st.success(f"PCA Completed, with {n_com} components")
        # st.write(f"Explained Variance: {round(np.sum(pca.explained_variance_ratio_), 4) * 100}%")
        
        # pca_df = pd.DataFrame(data = tpca, columns = [f"PC{i}" for i in range(1, n_com + 1)])
        # return pca_df
    else:
        with open('PCA.pkl', 'rb') as file:
            R_pca = pickle.load(file)
            
        n_pca = R_pca.transform(df)
        st.success(f"PCA Completed, with {n_com} components")
        n_pca_df = pd.DataFrame(data = n_pca, columns = [f"PC{i}" for i in range(1, n_com + 1)])
        return n_pca_df
    
    pass
def encode_column(data_frame, column_name):
    data_frame[column_name] = data_frame[column_name].map({'M': 1, 'B': 0})
    return data_frame


def upsample_data(data_frame, target_column):
    minority_samples = data_frame[data_frame[target_column] == 1]
    majority_samples = data_frame[data_frame[target_column] == 0]

    upsampled_minority = resample(
        minority_samples, replace=True, n_samples=len(majority_samples), random_state=42)

    upsampled_data = pd.concat([upsampled_minority, majority_samples])
    upsampled_data = upsampled_data.reset_index(drop=True)

    return upsampled_data

def add_parameter(clf_name):
    params = dict()
    if clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    if clf_name == "Random Forest":
        MD = st.sidebar.slider("max_depth", 1, 15)
        params["max_depth"] = MD
    if clf_name == "Decision Tree":
        MD = st.sidebar.slider("max_depth", 1, 15)
        params["max_depth"] = MD
    if clf_name == "Naive Bayes":
        pass
    if clf_name == "Logistic Regression":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    return params

def get_classifier(clf_name, params):
    clf = None
    if clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(max_depth=params["max_depth"])
    elif clf_name == "Decision Tree":
        clf = tree.DecisionTreeClassifier(max_depth=params["max_depth"])
    elif clf_name == "Naive Bayes":
        clf = GaussianNB()
    elif clf_name == "Logistic Regression":
        clf = LogisticRegression(C=params["C"])
    return clf


def model_building():
    render_subtitle("Model Building")
    
    df = pd.read_csv("Cancer_Prediction.csv")
    
    df = encode_column(df, 'diagnosis')
    
    df = upsample_data(df, 'diagnosis')
    
    X = df.drop('diagnosis', axis=1)
    y = df["diagnosis"]
    
    st.write("")
    if st.checkbox("Display Sample Data"):
        st.dataframe(df.head())
    
    st.write("")
    if st.checkbox("Start Preprocessing"):
        PCA_Component = st.sidebar.slider("PCA Component", 1, 30)
        if PCA_Component > 1:
            X = pre_processing(X, PCA_Component, 1)
            st.success("Preprocessed successfully")
            
            if st.checkbox("Build Model"):
                st.write("Please select seed value for reproducibility")
                st.write("Select Classifier")
                st.write("Select Parameters Turning")
                seed = st.sidebar.slider("Seed Value", 0, 200)
                classifier_name = st.sidebar.selectbox("Select Classifier", (None, "SVM", "KNN", "Random Forest", "Decision Tree", "Naive Bayes", "Logistic Regression"))
                params = add_parameter(classifier_name)
                
                if st.checkbox("Seed Value , Classifier Selected and Parameters Selected"):
                    if seed > 0 and classifier_name is not None:
                        st.success(f"Seed Value Selected: {seed}")
                        st.success(f"Classifier Selected: {classifier_name}")
                        st.success(f"Parameters Selected: {params}")
                        st.write("")
                    
                        
                    if st.checkbox("Fit Model"):
                        clf = get_classifier(classifier_name, params)
                    
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        acc = metrics.accuracy_score(y_test, y_pred)
                        f1_score = metrics.f1_score(y_test, y_pred)
                        precision = metrics.precision_score(y_test, y_pred)
                        st.success("Training Completed")
                    
                else:
                    st.error("Seed Value must be greater than 0")
                    st.error("Please select a Classifier")
        else:
            st.warning("No Preprocessing Done")
            st.error("Please select more than 1 component")  

    
        #st.warning("Kindly Predict using all the Model before testing the model on New Data")
        
        if st.checkbox("Predict"):
            #Loading Pickle Files:
            model_save_path = f"{classifier_name}.pkl"
            with open(model_save_path,'wb') as file:
                pickle.dump(clf, file)
            
            #st.success(f"Classifier = {classifier_name}")
            st.balloons()
            st.write(f"Accuracy Score = {round(acc *100, 2)}%")
            st.write(f"F1 Score = {round(f1_score *100, 2)}%")
            st.write(f"Precision Score = {round(precision *100, 2)}%")
        
            st.write("")
            create_heading('Confusion Matrix',2)
        
            cm = metrics.confusion_matrix(y_test, y_pred, labels=[1,0])
            df_cm = pd.DataFrame(cm, index=[i for i in [1, 0]], columns=[i for i in ['Predicted 1(Malignant)', 'Predicted 0(Benign)']])
            sns.heatmap(df_cm, annot=True, fmt=".3g", cbar=False)
            plt.title(f'{classifier_name} Classifier Confusion Matrix')
            st.pyplot(plt)

def prediction():
    render_subtitle("Prediction")
    st.write("")
    
    st.warning("Kindly Predict using only the Previously Trained Models")
    
    #data = st.file_uploader("Upload Dataset:", type=['csv','txt'])
    st.write("Synthetic Unseen Data")
    st.write("This data was randomly generated, so as to test the accuracy of the model, The Data Mimics the real dataset")
    
    data = pd.read_csv("synthetic_data.csv")
    
    st.warning("PCA N Component = 15")
    pro_df = pre_processing(data,n_com=15, pca=2) 
    
    if st.checkbox("Display Sample Data"):
        st.dataframe(data.head())
        
    # if st.checkbox("Check for Null Values"):
    #     st.write(pro_df.isnull().sum())
    
    st.write("")
    
    option = st.sidebar.selectbox("Select your preferred Model", ("SVM", "KNN", "Random Forest", "Decision Tree", "Naive Bayes", "Logistic Regression"))
    
    if st.button("Predict"):
        if option == "SVM":
            with open('SVM.pkl', 'rb') as file:
                model = pickle.load(file)
            st.success("Support Vector Machine Classifier Predictions")
        if option == "KNN":
            with open('KNN.pkl', 'rb') as file:
                model = pickle.load(file)
            st.success("K-Nearest Neighbor Classifier Predictions")
        if option == "Random Forest":
            with open('Random Forest.pkl', 'rb') as file:
                model = pickle.load(file)
            st.success("Random Forest Classifier Predictions")
        if option == "Decision Tree":
            with open('Decision Tree.pkl', 'rb') as file:
                model = pickle.load(file)
            st.success("Decision Tree Classifier Predictions")
        if option == "Naive Bayes":
            with open('Naive Bayes.pkl', 'rb') as file:
                model = pickle.load(file)
            st.success("Naive Bayes Classifier Predictions")
        if option == "Logistic Regression":
            with open('Logistic Regression.pkl', 'rb') as file:
                model = pickle.load(file)
            st.success("Logistic Regression Classifier Predictions")
        
        prediction = model.predict(pro_df)
        #Concat into a dataframe
        predicted_df = pd.concat([data["id"], pd.Series(prediction, name="Result")],axis=1)
        predicted_df = predicted_df.reset_index(drop=True)
        # predicted_df.columns = ["id","Result"]
        predicted_df['Result_Label'] = predicted_df["Result"].apply(lambda x: 'Malignant' if x == 1 else 'Benign')
            
            # if prediction[20] == 1:
            #     st.error("Tumor Detected")
            # else:
            #     st.success("Tumor Not Detected")
        st.dataframe(predicted_df)
            


def about_us():
    render_subtitle("About Us")
    
    lottie_about = load_lottiefile("about.json")
    st_lottie(
        lottie_about,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        #renderer="svg", # canvas
        height=None,
        width=None,
        key=None,
    )
    
    st.balloons()

# def main():
#     activities = ['Home', 'EDA', 'Visualization', 'Model Building', 'Prediction', 'About Us']
#     option = st.sidebar.selectbox('Select Activity', activities)
    
    
#     if option == 'Home':
#         home()
#     elif option == 'EDA':
#         eda()
#     elif option == 'Visualization':
#         visualization()
#     elif option == 'Model Building':
#         model_building()
#     elif option == 'Prediction':
#         prediction()
#     elif option == 'About Us':
#         about_us()


# if __name__ == '__main__':
#     main()


def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu", #Remove the default title
            options=["Home", "EDA", "Visualization", "Model Building", "Prediction", "About Us"], #Required
            icons=["house", "bar-chart", "graph-up", "pencil-square", "clipboard-data", "person"], #Required
            menu_icon="cast", #Optional
            default_index=0,
            #orientation="horizontal"
            )
    if selected == "Home":
        home()
    if selected == "EDA":
        eda()
    if selected == "Visualization":
        visualization()
    if selected == "Model Building":
        model_building()
    if selected == "Prediction":
        prediction()
    if selected == "About Us":
        about_us()


if __name__ == "__main__":
    main()

  




