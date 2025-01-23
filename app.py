import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import requests
from io import BytesIO
import time

def load_data():
    try:
        # Load test data for seeing current image
        test_data = pickle.load(open('test_data.pkl','rb'))
        test_data_ = pd.DataFrame(test_data)
        
        # Load train data for seeing recommendations
        train_data = pickle.load(open('img_data.pkl','rb'))
        train_data_ = pd.DataFrame(train_data)
        
        # Load model
        knn = pickle.load(open('model_recommend.pkl','rb'))
        
        # Load TF-IDF test array
        X_test = pickle.load(open('test_array.pkl','rb'))
        
        return test_data_, train_data_, knn, X_test
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def home_page():
    st.title("Recommendation System for Online Shopping 🦋 ✨")
    
    st.sidebar.image("images.jpg", caption="Recommendation System for Online Shopping", width=300)
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **Developer:** Gangadevi M
                    
    **College:** Sri Vijay Vidyalaya College of Arts and Science
    
    This application is built using Streamlit.
    """)
    
    st.markdown("""
    ## Project Overview
    
    This recommendation system is designed to enhance the online shopping experience by providing personalized and intelligent product suggestions for H&M clothing items.
    
    ### Key Features:
    - Intelligent product recommendations
    - Based on similarity in title, brand, and color
    - Uses K-Nearest Neighbors algorithm
    
    ### How it Works:
    1. Select a product from the dropdown
    2. View product details
    3. Click 'Get Recommendations' to see similar items
    """)
    


def recommendation_page():
    st.title("Fashion Recommendation System")
    
    st.header('About Recommendation Model:')
    st.markdown(
        "The model uses 'Nearest Neighbours' to find similar products. "
        "Recommendations depend on product title, color, and brand."
    )
    
    # Load data
    test_data_, train_data_, knn, X_test = load_data()
    
    if test_data_ is None:
        return
    
    # Product selection
    title_current = st.selectbox('Search for the product:', 
                    list(test_data_['title']))
    product = test_data_[(test_data_['title'] == title_current)]
    s1 = product.index[0]
    
    # Display current product details
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(test_data_['medium_image_url'].values[s1])
    with c2:
        st.text('Brand--->')
        st.text('Color--->')
        st.text('Price in $--->')
    with c3:
        st.text(test_data_['brand'].values[s1])
        st.text(test_data_['color'].values[s1])
        st.text(test_data_['formatted_price'].values[s1])
    
    # Get recommendations
    distances, indices = knn.kneighbors([X_test.toarray()[s1]])
    result1 = list(indices.flatten())[:5]
    result2 = list(indices.flatten())[5:]
    
    if st.button('Get Recommendations'):
        st.success('Recommendations for you:')
        
        # First row of recommendations
        col1, col2, col3, col4, col5 = st.columns(5)
        lst1 = [col1, col2, col3, col4, col5]
        for i, j in zip(lst1, result1):
            with i:
                st.text(train_data_['brand'].values[j])
                st.text(train_data_['color'].values[j])
                st.image(train_data_['medium_image_url'].values[j])
        
        # Second row of recommendations
        col6, col7, col8, col9, col10 = st.columns(5)
        lst2 = [col6, col7, col8, col9, col10]
        for k, l in zip(lst2, result2):
            with k:
                st.text(train_data_['brand'].values[l])
                st.text(train_data_['color'].values[l])
                st.image(train_data_['medium_image_url'].values[l])
        
        st.success('Thank You for Shopping!')

def exit_page():
    st.title("Thank You for Using Fashion Recommendation System 👋")
    
    st.balloons()
    
    st.markdown("""
    ### Hope You Had a Great Shopping Experience!
    
    #### Quick Feedback
    - Did you find the recommendations helpful?
    - Any suggestions for improvement?
    
    Feel free to contact the developer for any queries.
    """)
    
    # Countdown timer with compatibility fix
    progress_text = "Closing application in..."
    my_bar = st.progress(0)
    
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    
    st.success("Application closed. Thank you!")

def main():
    st.set_page_config(page_title="Fashion Recommendation System", page_icon="👗")
    
    # Create a sidebar navigation
    page = st.sidebar.radio("Navigate", ["Home", "Recommendations", "Exit"])
    
    # Hide Streamlit style
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer { 
        content: Made by Bairagi Saurabh :); 
        visibility: visible; 
        display: block; 
        position: relative; 
        padding: 5px; 
        top: 2px; 
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    if page == "Home":
        home_page()
    elif page == "Recommendations":
        recommendation_page()
    else:
        exit_page()

if __name__ == "__main__":
    main()