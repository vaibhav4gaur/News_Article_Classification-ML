import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st


# Streamlit page configuration
st.set_page_config(page_title="News Article Classification", layout="wide")

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load and process data
@st.cache_data
def load_and_process_data(sample_size=1000):
    dataframe_fake = pd.read_csv("Fake.csv")
    dataframe_true = pd.read_csv("True.csv")
    dataframe_fake['label'] = 1
    dataframe_true['label'] = 0
    news_dataframe = pd.concat([dataframe_fake.sample(sample_size, random_state=42), dataframe_true.sample(sample_size, random_state=42)], axis=0).reset_index(drop=True)
    return news_dataframe

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stem_content(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

# Vectorize data
@st.cache_data
def vectorize_data(news_dataframe):
    news_dataframe['content'] = news_dataframe['text'].apply(stem_content)
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(news_dataframe['content'])
    y = news_dataframe['label'].values
    return X, y, vectorizer

# Train model
@st.cache_data
def train_model(_X, y):
    X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# Load data and train model
news_dataframe = load_and_process_data()
X, y, vectorizer = vectorize_data(news_dataframe)
model, accuracy = train_model(X, y)

# Streamlit app
st.title('News Article Classification')
st.write("""
    Welcome to the News Article Classification! This application uses Natural Language Processing (NLP) techniques to classify news articles as real or fake. 
    click on 'Predict' to see the result.
""")

# Input form
input_text = ""
# uploaded_file = None
with st.form(key='news_form'):
    input_text = st.text_area('📝 Enter News Article', height=250)
    submit_button = st.form_submit_button(label='Predict')
   




# Only perform prediction when submit button is clicked
if submit_button:
    if input_text.strip():
        with st.spinner('Analyzing the article...'):
            processed_input = stem_content(input_text)
            input_data = vectorizer.transform([processed_input])
            pred = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0][pred]
            result = 'The News is **Fake**' if pred == 1 else 'The News is **Real**'
            st.success(result)

          
    else:
        st.error("Please enter some text to analyze.")


