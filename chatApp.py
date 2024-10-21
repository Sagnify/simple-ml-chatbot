import tkinter as tk
from tkinter import scrolledtext
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
import wikipedia
import geocoder
import re
import math_exp_processor as math

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

# Initialize user context storage
user_context = {}

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
    current_time = now.strftime("%I:%M:%S %p")
    return f"The current time is {current_time}"

def get_current_date():
    now = datetime.now()
    current_date = now.strftime("%B %d, %Y")  # Month Day, Year format
    return f"Today's date is {current_date}"

# Function to get a Wikipedia summary
def get_wikipedia_summary(user_input):
    query = user_input.replace("who is ", "").replace("define ", "").replace("tell me about ", "").replace("what is ", "").strip()
    try:
        page_summary = wikipedia.summary(query, sentences=2)
        return page_summary[:500]
    except wikipedia.exceptions.DisambiguationError as e:
        suggestion = e.options[0] if e.options else "no suggestions"
        return f"Sorry, I couldn't find any information on that. Did you mean '{suggestion}'? (yes/no)"
    except wikipedia.exceptions.PageError:
        search_results = wikipedia.search(query)
        if search_results:
            closest_match = search_results[0]
            return f"Sorry, I couldn't find any information on that. Did you mean '{closest_match}'? (yes/no)"
        else:
            return "Sorry, I couldn't find any information on that."

# Function to get weather details using WeatherAPI.com
def get_weather():
    city = user_context.get('city', 'Delhi')  # Use city from user context, default to 'Delhi'
    
    api_key = "eff4babce5af4b82896151346241310"  # Replace with your WeatherAPI key
    base_url = "http://api.weatherapi.com/v1/current.json"

    params = {
        'key': api_key,
        'q': city,
        'aqi': 'no'
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

# Function to get time-based greeting
def get_time_based_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good Morning"
    elif 12 <= current_hour < 18:
        return "Good Afternoon"
    else:
        return "Good Evening"
    
def get_current_city():
    # Get current location using geocoder
    g = geocoder.ip('me')
    return g.city if g.city else "Delhi"  # Default to 'Delhi' if city not found

def extract_city(user_input):
    # Use regex to find words in the input, filtering out symbols
    words = re.findall(r'\b\w+\b', user_input)  # Find all words

    # Check for the presence of 'in' or 'at'
    if 'in' in words or 'at' in words:
        # Split the input on 'in' or 'at' and take the last part as the city
        split_input = re.split(r'\bin\b|\bat\b', user_input, flags=re.IGNORECASE)
        if len(split_input) > 1:
            # Get the last part, filter out non-alphabetic characters
            city_candidates = re.findall(r'\b\w+\b', split_input[-1])
            city = city_candidates[0] if city_candidates else None
            return city if city else get_current_city()

    # If 'in' or 'at' is not found, fallback to current location
    return get_current_city()


# Function to get a response based on the predicted intent
def get_response(predicted_class, user_input):
    # Check for weather request
    if predicted_class == 'weather':
        user_context['city'] = extract_city(user_input)  # Assume the last word is the city name
        return get_weather()
    
    if predicted_class == 'time':
        return get_current_time()
    
    if predicted_class == 'date':
        return get_current_date()

    if predicted_class == 'knowledge':
        return get_wikipedia_summary(user_input)

    # Analyze sentiment
    if predicted_class == 'sentiment':
        sentiment_score = analyze_sentiment(user_input)
        if sentiment_score > 0:
            user_context['last_sentiment'] = "positive"
            return "I'm really happy to hear that you're feeling positive! What brought on this good mood?"
        elif sentiment_score < 0:
            user_context['last_sentiment'] = "negative"
            return "I'm sorry to hear that you're feeling down. It's completely okay to have these feelings. If you'd like to share, I'm here to listen."
        else:
            user_context['last_sentiment'] = "neutral"
            return "It sounds like you’re feeling a bit indifferent. Anything in particular on your mind?"

    # Generate follow-up responses based on last sentiment
    last_sentiment = user_context.get('last_sentiment')
    user_declined = user_context.get('user_declined_talk', False)  # Track if user declined to talk

    # If user explicitly says "no," record their preference
    if predicted_class in ['no', 'not_in_mood', 'talk_later']:
        user_context['user_declined_talk'] = True  # Update the flag on decline
        # Check if user has previously shared a sentiment
        if last_sentiment in ['positive', 'negative']:
            return "That's completely fine! We can talk about something else if you'd like."
        else:
            return "I understand. We can change the subject. What else would you like to discuss?"

    if last_sentiment == 'positive':
        if user_declined:
            return "That’s great to hear! If you want, we can chat about something else. What would you like to discuss?"
        return "It’s wonderful that you’re feeling good! What’s been going well for you?"
    
    if last_sentiment == 'negative':
        if user_declined:
            return "I understand. If you prefer, we can talk about something else. What interests you?"
        return "I know it can be tough sometimes. Would you like to talk more about what's troubling you, or should we change the topic?"

    if last_sentiment == 'neutral':
        return "I’m here for whatever you want to chat about. Is there something specific on your mind?"
    
    # If the predicted class is 'math', extract and evaluate the expression
    if predicted_class in ['add', 'subtract', 'divide', 'multiply']:
        # Use handle_math_response to classify and evaluate the math expression
        expression = math.classify_and_convert(user_input, predicted_class)
        if expression and "Invalid" not in expression:
            result = math.evaluate_expression(expression)
            print(f"The result of {expression} is {result}")
            return f"The result of {expression} is {result}"
        else:
            print(expression)



    # Fallback for other intents
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])
            return response

    return "I'm sorry, I didn't understand that."





# GUI Components
def handle_user_input(event=None):
    user_input = user_input_text.get()
    chat_window.config(state=tk.NORMAL)  # Enable editing
    chat_window.insert(tk.END, f"You: {user_input}\n")
    user_input_text.delete(0, tk.END)

    predicted_class = predict_intent(user_input)
    user_context['predicted_class'] = predicted_class 
    response = get_response(predicted_class, user_input)

    chat_window.insert(tk.END, f"Bot: {response}\n")
    chat_window.config(state=tk.DISABLED)  # Disable editing again
    update_user_details()

def update_user_details():
    details_text.delete(1.0, tk.END)  # Clear current text
    details_text.insert(tk.END, json.dumps(user_context, indent=4))  # Display user context

# Create the main window
window = tk.Tk()
window.title("Chatbot")

# Create a text area for the chat
chat_window = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=20, state='disabled')
chat_window.grid(row=0, column=0, columnspan=2)

# Create a text box for user input
user_input_text = tk.Entry(window, width=40)
user_input_text.grid(row=1, column=0)

# Create a button to send the message
send_button = tk.Button(window, text="Send", command=handle_user_input)
send_button.grid(row=1, column=1)

# Create a text area to display user details
details_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=30, height=20)
details_text.grid(row=0, column=2, rowspan=2)

# Bind the Enter key to send the message
window.bind('<Return>', handle_user_input)

# Start the GUI event loop
window.mainloop()