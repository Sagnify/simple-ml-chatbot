import nltk
import numpy as np
import json
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import random
import requests
from datetime import datetime
from textblob import TextBlob

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
import wikipedia
import difflib


from math_exp_processor import convert_to_expression, evaluate_math_expression, process_math_input

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.keras')

# Load the intents file
with open('intents.json') as data_file:
    intents = json.load(data_file)

# Load the words and classes
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess user input
def preprocess_input(user_input):
    tokens = nltk.word_tokenize(user_input)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]

    bag = [0] * len(words)
    for token in tokens:
        if token in words:
            bag[words.index(token)] = 1

    return np.array(bag)

# Function to predict the intent of the user input
def predict_intent(user_input):
    bag_of_words = preprocess_input(user_input)
    prediction = model.predict(np.array([bag_of_words]))
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    return predicted_class

# Function to analyze sentiment
def analyze_sentiment(user_input):
    analysis = TextBlob(user_input)
    return analysis.sentiment.polarity
    

# Function to get the current time
def get_current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return f"The current time is {current_time}"

def get_wikipedia_summary(user_input):
    # Clean the user input for the query
    query = user_input.replace("who is ", "").replace("define ", "").replace("tell me about ", "").replace("what is ", "").strip()

    try:
        # Search for the page using the wikipedia library
        page_summary = wikipedia.summary(query, sentences=2)  # Get a summary of the page
        return page_summary[:500]  # Limiting the summary to 500 characters
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation by suggesting the first option
        suggestion = e.options[0] if e.options else "no suggestions"
        return f"Sorry, I couldn't find any information on that. Did you mean '{suggestion}'? (yes/no)"
    except wikipedia.exceptions.PageError:
        # Suggest alternative titles if the page does not exist
        search_results = wikipedia.search(query)  # Search for related pages
        if search_results:
            closest_match = search_results[0]  # Suggest the closest match
            return f"Sorry, I couldn't find any information on that. Did you mean '{closest_match}'? (yes/no)"
        else:
            return "Sorry, I couldn't find any information on that."



# Function to get weather details using WeatherAPI.com
def get_weather(city):
    api_key = "eff4babce5af4b82896151346241310"  # Replace with your WeatherAPI key
    base_url = "http://api.weatherapi.com/v1/current.json"
    
    params = {
        'key': api_key,
        'q': city,
        'aqi': 'no'  # 'no' to disable air quality index information
    }
    
    response = requests.get(base_url, params=params)
    weather_data = response.json()
    
    if 'current' in weather_data:
        current = weather_data['current']
        condition = current['condition']['text']
        temperature = current['temp_c']
        return f"The weather in {city} is currently {condition} with a temperature of {temperature}°C."
    else:
        return "I couldn't retrieve the weather information right now."
    
def get_time_based_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good Morning"
    elif 12 <= current_hour < 18:
        return "Good Afternoon"
    else:
        return "Good Evening"


# Function to get a response based on the predicted intent
def get_response(predicted_class, user_input):
    # If the predicted class is 'weather' or 'time', call the corresponding API
    if predicted_class == 'weather':
        print("Bot: Can you please enter your city")
        city = input("You: ")
        return get_weather(city)
    
    if predicted_class == 'time':
        return get_current_time()
    
    # Check if the intent is a knowledge
    if predicted_class == 'knowledge':
        return get_wikipedia_summary(user_input)
    
    if predicted_class == 'sentiment':
        sentiment_score = analyze_sentiment(user_input)
        if sentiment_score > 0:
            return "I'm glad to hear you're feeling positive!"
        elif sentiment_score < 0:
            return "I'm sorry to hear that. If you want to talk about it, I'm here for you."
        else:
            return "It seems like you're feeling neutral. Anything specific on your mind?"

    # Check if the intent is a greeting
    is_greeting = predicted_class == "greeting"
    
    # Get time-based greeting if the intent is a greeting
    time_greeting = get_time_based_greeting() if is_greeting else ""
    
    # Otherwise, return a random response from the intents file
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])

            # If the predicted class is 'math', extract and evaluate the expression
            if predicted_class == 'math':
                expression = process_math_input(user_input)
                if expression:
                    print(f"Converted expression: {expression}")
                    result = evaluate_math_expression(expression)
                    print(f"Result: {result}")
                else:
                    print("Sorry, I couldn't understand that expression.")

            # Only append the time-based greeting for greeting intent
            if is_greeting:
                return f"{time_greeting}! {response}"
            else:
                return response

    return "I'm sorry, I didn't understand that."

print("Chatbot is running! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    predicted_class = predict_intent(user_input)
    response = get_response(predicted_class, user_input)
    print("Bot:", response)

    if 'Did you mean' in response:
        user_response = input("You: ")
        if user_response.lower() == 'yes':
            # Extract the suggested title from the response
            suggested_title = response.split("Did you mean ")[1].split("?")[0].strip()  # Ensure we get the right part
            print(f"Fetching summary for: {suggested_title}")
            # Call the function with the suggested title
            new_response = get_wikipedia_summary(suggested_title)
            print(new_response)
        else:
            print("Okay, let me know if you have another question.")
