import re
from sympy import sympify, SympifyError

# Mapping for number words to digits
number_mapping = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90"
}

# Function to convert number words to digits
def convert_word_to_number(word):
    return number_mapping.get(word.lower(), word)

# Function to extract numbers from user input
def extract_numbers(user_input):
    # Split the input and convert words to numbers
    words = user_input.split()
    numbers = []
    for word in words:
        if word.isdigit():
            numbers.append(word)
        else:
            number = convert_word_to_number(word)
            if number.isdigit():
                numbers.append(number)
    return numbers

# Define function to classify and convert the user input into a mathematical expression
def classify_and_convert(user_input, predicted_class):
    # Match the predicted class to determine which operation to convert
    if predicted_class == "add":
        return convert_addition(user_input)
    elif predicted_class == "subtract":
        return convert_subtraction(user_input)
    elif predicted_class == "multiply":
        return convert_multiplication(user_input)
    elif predicted_class == "divide":
        return convert_division(user_input)
    else:
        return "I'm sorry, I couldn't understand that."

# Function to convert addition input into a mathematical expression
def convert_addition(user_input):
    numbers = extract_numbers(user_input)
    if len(numbers) == 2:
        expression = f"{numbers[0]} + {numbers[1]}"
        return expression
    return "Invalid addition input."

# Function to convert subtraction input into a mathematical expression
def convert_subtraction(user_input):
    numbers = extract_numbers(user_input)
    if len(numbers) == 2:
        expression = f"{numbers[0]} - {numbers[1]}"
        return expression
    return "Invalid subtraction input."

# Function to convert multiplication input into a mathematical expression
def convert_multiplication(user_input):
    numbers = extract_numbers(user_input)
    if len(numbers) == 2:
        expression = f"{numbers[0]} * {numbers[1]}"
        return expression
    return "Invalid multiplication input."

# Function to convert division input into a mathematical expression
def convert_division(user_input):
    numbers = extract_numbers(user_input)
    if len(numbers) == 2:
        expression = f"{numbers[0]} / {numbers[1]}"
        return expression
    return "Invalid division input."

# Function to evaluate the mathematical expression using sympy
def evaluate_expression(expression):
    try:
        result = sympify(expression)
        return str(result)
    except SympifyError:
        return "Sorry, I can't compute that."
    
# Example usage
# if __name__ == "__main__":
#     user_input = "divide 4 with 2"  # Change this input to test different cases
#     predicted_class = "divide"  # Simulate the output from your model

#     expression = classify_and_convert(user_input, predicted_class)
#     if expression and "Invalid" not in expression:
#         result = evaluate_expression(expression)
#         print(f"The result of {expression} is {result}")
#     else:
#         print(expression)