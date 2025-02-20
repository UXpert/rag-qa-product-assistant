import openai
import os
from dotenv import load_dotenv

# Load the .env file to access the API key
load_dotenv()

# Set the OpenAI API key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if the key loaded successfully
if not openai.api_key:
    print("❌ API key not found. Make sure it's in your .env file.")
else:
    print("✅ API key loaded successfully!")

# Load the document content
def load_document(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        print("✅ Document loaded successfully!")
        return content
    except FileNotFoundError:
        print("❌ Document not found. Check the file path.")
        return ""

# Load the knowledge.txt file from the documents folder
document_content = load_document("documents/knowledge.txt")

import re

# Function to remove common filler words (stop words)
def clean_and_split(text):
    stop_words = {"what", "is", "the", "a", "an", "of", "and", "in", "on", "with", "for", "to"}
    words = re.findall(r'\w+', text.lower())
    return [word for word in words if word not in stop_words]

# Improved function to find relevant sentences
def find_relevant_sentences(question, document):
    sentences = re.split(r'(?<=[.!?]) +', document)
    question_words = set(clean_and_split(question))
    best_match = ""
    best_score = 0

    for sentence in sentences:
        sentence_words = set(clean_and_split(sentence))
        common_words = question_words.intersection(sentence_words)
        score = len(common_words)

        if score > best_score:
            best_score = score
            best_match = sentence

    return best_match if best_match else "No relevant sentence found."

import openai

# Function to generate an answer using OpenAI
def generate_answer_with_openai(question, context):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Use gpt-3.5-turbo for faster responses and affordability
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions using the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\nAnswer:"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating answer: {e}"
    
# Ask the user a question
user_question = input("🤔 Enter your question: ")

# Find the most relevant sentence from the document
relevant_sentence = find_relevant_sentences(user_question, document_content)
print(f"🔎 Relevant sentence: {relevant_sentence}")

# Generate an answer using OpenAI
final_answer = generate_answer_with_openai(user_question, relevant_sentence)
print(f"🤖 AI-generated answer: {final_answer}")