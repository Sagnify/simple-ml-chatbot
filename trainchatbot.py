import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random

import tensorflow as tf


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!', '@', '$', 'to', 'the']

# Load the intents file
with open('intents.json') as data_file:
    intents = json.load(data_file)

# Process intents and patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # Tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        words.extend(w)  # Add to our words list
        documents.append((w, intent['tag']))  # Add to documents

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word, remove duplicates and ignore certain characters
words = [lemmatizer.lemmatize(w.lower()) for w in words if w.lower() not in ignore_words]
words = sorted(list(set(words)))

# Sort classes as well
classes = sorted(list(set(classes)))

# Save the words and classes to pickle files
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Initializing training data
training = []
output_empty = [0] * len(classes)

# Create the training set, a bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]

    # Lemmatize each word in the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create the bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data and convert to a NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)
# Split into train_x (inputs) and train_y (outputs)
train_x = np.array(list(training[:, 0]), dtype=float)
train_y = np.array(list(training[:, 1]), dtype=float)

print("Training data created")
print("train_x: ", train_x)
print("train_y: ", train_y)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(len(train_y[0]), activation = 'softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9, nesterov =True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras')

print("Model created and saved")
