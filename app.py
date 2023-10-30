import streamlit as st
import pandas as pd
import joblib
import datetime
import base64  # Import base64 library

# Load the trained SVM model and TF-IDF vectorizer
model = joblib.load("svm_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define class labels
class_labels = {0: "Figurative", 1: "Irony", 2: "Regular", 3: "Sarcasm"}

# Define custom CSS styles for the background image
background_style = """
<style>
body {
    background-image: url("YOUR_IMAGE_URL_OR_PATH");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""

# Page 1: Model Description
def model_description():
    st.write("In this project, We embarked on a comprehensive analysis of textual data, specifically focusing on tweets extracted from various sources. The core objective was to unveil the underlying sentiments and nuanced expressions within these tweets. To achieve this, I employed state-of-the-art sentiment analysis techniques, harnessing the power of natural language processing and machine learning.")
    st.write("The dataset at the heart of this endeavor was structured with two essential columns: 'tweets' and 'classes', comprising figurative, irony, sarcasm, and regular categories. Leveraging the predictive capabilities of my model, I successfully assigned each tweet to its corresponding class, providing insights into the intricate emotions and rhetorical styles embedded within the Twitterverse.")
    st.write("This project delved into the intricate world of sentiment, discerning not only basic positive and negative sentiments but also the more sophisticated aspects of figurative language, irony, and sarcasm. By doing so, it offered a deeper understanding of the diverse and often playful nature of human expression on social media. The results of this analysis hold the potential for invaluable applications in fields such as social media marketing, opinion mining, and trend tracking, making it a significant contribution to the realm of sentiment analysis.")
    # ... (Rest of your code for model_description)

# Page 2: Sentiment Prediction
def sentiment_prediction():
    st.subheader("Sentiment Prediction")

    # Input for single statement
    user_input = st.text_input("Enter a statement:")
    if st.button("Predict"):
        # Vectorize the input text
        input_vector = tfidf_vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(input_vector)[0]
        sentiment = class_labels.get(prediction, "Unknown")

        # Display result
        st.write(f"Predicted Sentiment: {sentiment}", color="black")

    # Upload CSV file for batch prediction
    st.subheader("Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Assuming the CSV file has a 'text' column containing statements
            if 'text' in df.columns:
                # Perform TF-IDF vectorization on the statements
                X_test_tfidf = tfidf_vectorizer.transform(df['text'])
                # Predict sentiments
                df['predicted_sentiment'] = model.predict(X_test_tfidf)
                
                # Map numeric predictions to labels
                df['predicted_sentiment'] = df['predicted_sentiment'].apply(lambda x: class_labels.get(x, "Unknown"))

                # Display the predictions
                st.dataframe(df)
                
                # Create a download link for the predictions
                today = datetime.date.today()
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_predictions_{today}.csv">Download Predictions as CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("CSV file must contain a 'text' column.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Streamlit app
def main():
    # Apply the custom CSS styles for the background image
    st.set_page_config(page_title="Tweets Sentimental Predicition",page_icon='https://img.icons8.com/office/16/twitter.png')
    st.markdown("""
        <style>
            /* Set background image */
            .stApp {
                background-image: url("https://media.cnn.com/api/v1/images/stellar/prod/120127055441-twitter.jpg?q=x_0,y_109,h_765,w_1360,c_crop/h_720,w_1280");
                background-attachment: fixed;
                background-size: cover;
            }
            
            /* Center the title */
            .title {
                text-align: center;
                border-radius: 50px;
                color: rgb(255 255 255);
                background: #0894ca8f;

            }
            
            /* Increase the size of text inputs */
            .stNumberInput label p{
                padding: 10px;
                font-size: 20px;
           }

           /* Increase the size of number inputs */
           .stNumberInput input[type="number"] {
               padding: 10px;
               font-size: 18px;
               }
           
           /* Prediction output  */ 
           .css-nahz7x p{
                font-size: 18px   
            }
           
           .st-b7 {
               color: black;
               font-size: 22px
           }
           
           .st-ch {
               background-color: rgb(255 255 255 / 70%) !important;
           }
           
           /* Upload CSV design */
           
           .stMarkdown{
               margin-top: 35px;
               margin-bottom: 25px;

               }
                
           #upload-csv-file{
               text-align: center;
               color: #dcdcdc;
               font-size: 32px;
               background: #31c0238f;
               border-radius: 50px;
               padding: 8px;
               margin-bottom: 25px;
           }
                
            .stAlert{
                background: rgb(255 255 255 / 65%);
                border-radius: 15px;
                }
                
         /* Input label color */
         .css-1qg05tj{
             color: rgb(164 209 234);
             }
        </style>
        <h1 class="title">Tweets Sentiment Analysis</h1>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    pages = {
        "Model Description": model_description,
        "Sentiment Prediction": sentiment_prediction,
    }
    page = st.sidebar.radio("Go to:", list(pages.keys()))
    pages[page]()

if __name__ == "__main__":
    main()
