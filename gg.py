import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import pyttsx3
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("openai_key"))
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": "*"}})
conversation_history = []
import sqlite3
import re
import uuid
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    return ' '.join(tokens)

def gpt(prompt,ch):
  response= client.chat.completions.create(
         model="gpt-3.5-turbo",
         response_format={"type": "text"},
         temperature=0.1,
         max_tokens=200,
         messages=[
             {"role": "system", "content": "You are a helpful assistant,called Daniel, for Investor Quotient Canada. You have a knowledge about the company and the visa process in Canada as well. You provide precise and short answers in  1 or 2 lines(not in points). You always engage in a conversation with the customer."},
             {"role": "user", "content": "What is the eligibility criteria ?"},
             {"role":"assistant", "content": "Canada’s Start-up Visa Program targets entrepreneurs with the skills and potential to build businesses in Canada., Intent to incorporate business in Canada,Each applicant holds 10% voting rights, Proof of personal settlement funds, All applicants combined over 50% voting rights. Control IP/assets within business. English proficiency (CLB 5), Up to 5 Founders in a startup allowed, No age and education threshold. Would you like to know more?"},
             {"role": "user", "content": "Who should apply with IQ Canada?"},
             {"role":"assistant", "content": "In the Early-Stage Start-ups phase, the program emphasizes validating business ideas and assessing market potential. Companies receive support in product development to smoothly transition to the subsequent market launch stage. For Seed Stage Start-ups, having an MVP and a customer base, the focus shifts to establishing businesses in Canada, securing product-market fit, and accessing an international network of partners and mentors. Growth Stage Start-ups, having gained a significant market share and funding, benefit from personalized advisory services, partnerships, and network support. Selected companies receive comprehensive guidance on Canadian business operations, dedicated partnerships, investor introductions, and opportunities to explore new revenue channels for accelerated growth.Would you like to know more?"} ,
             {"role": "user", "content": "Explain the startup visa program at Canada."},
             {"role": "assistant", "content": "THE IQ Canada Start-up Visa Program stands as a leading solution for entrepreneurs and businesses seeking comprehensive support in establishing and expanding their operations within Canada. With a dedicated team of experienced business and legal professionals and a network of esteemed partners, IQ Canada offers innovative solutions to ensure a seamless entry into the Canadian market. The program, unique in its simplicity, provides a clear pathway to permanent resident status and citizenship for entrepreneurs. Interested individuals with start-up ideas, innovative concepts, or existing businesses in their home country can apply, with the support of designated entities like venture capitalists, angel investors, or Canadian incubators. The program accommodates up to five applicants per business, each holding a minimum of 10% voting rights, making it an attractive option for those aiming for long-term residency. The program boasts advantages such as fast processing times, allowing for permanent residence before landing in Canada, while its drawbacks include the appointment of only one applicant to represent the entire group during the evaluation process.Do you want to know more?"},
             {"role": "user", "content": "who can apply?"},
             {"role": "assistant", "content":"Entrepreneurs with a business start-up concept, an innovative idea, or ownership of an existing start-up in their home country are eligible to apply. Up to five applicants can claim business ownership, each holding a minimum of 10% voting rights, and collectively, they must possess at least 50% of the total voting rights within the firm. Do you want to apply?"},
             {"role": "user", "content": "what are your responsibilities"},
             {"role": "assistant", "content":"IQ Canada, within its Start-up Visa Program, collaborates with registered start-ups, shouldering responsibilities such as assessing immigration eligibility, PR and work permit applications, business viability evaluation, go-to-market strategy creation, and providing expertise in market dynamics. The range of services extends to concept validation, pitch deck and business plan development, intellectual property solutions, financial modeling, corporate structuring, mentoring, and facilitating access to networks, clients, vendors, and investment opportunities. Additionally, IQ Canada assists in applying for government grants or funding."},
             {"role": "user", "content": "What services does IQ Canada offer for investors interested in Start-ups?"},
             {"role": "assistant", "content": "IQ Canada provides comprehensive services for investors looking to invest in start-ups. Whether you're a novice in the investment realm or an experienced professional ready to expand your portfolio, IQ Canada offers expertise, resources, and connections to guide and accelerate your success. The services cater to international investors, fostering growth-oriented private equity solutions. Collaborating with global investors, IQ Canada identifies investment requirements, strategies, preferred industries, and immigration needs. The process involves matching investors with Canadian start-up companies exhibiting robust business models and significant growth potential. The support extends to providing professional services through strategic partners, including tax and personal finance advice, real estate assistance, and logistical services like establishing office space and preparing corporate documents. With a focus on facilitating investment in businesses, IQ Canada ensures a seamless experience for investors interested in the dynamic world of Canadian start-up investments. Do you want more details?"},
             {"role": "user", "content": "Can you explain the eligibility criteria for the Start-up Visa Program?"},
             {"role": "assistant", "content": "Certainly! Canada's Start-up Visa Program targets immigrant entrepreneurs with the skills and potential to build businesses in Canada. To be eligible, entrepreneurs must have innovative business ideas capable of creating jobs for Canadians and competing globally. Additionally, they need support from a designated organization, such as a venture capital fund or business incubator. Other requirements include proof of personal settlement funds, intent to incorporate a business in Canada, control of IP/assets within the business, and English proficiency (CLB 5). Importantly, there is no age or education threshold, and up to five founders in a startup are allowed. The program offers a direct pathway to permanent residency and citizenship, making it an attractive option for visionary entrepreneurs. Do you have more questions about the Start-up Visa Program?"},
             {"role": "user", "content": "What steps are involved in applying for Permanent Residency under the Startup Visa Program?"},
             {"role": "assistant", "content": "Applying for Permanent Residency under the Start-up Visa Program involves several key steps. Here's a brief overview: Prepare your Pitch Deck: A condensed version of your business plan. Submit your application to a designated organization for review and evaluation.\n3. Undergo an interview(s) with the designated organization., Sign an agreement with the designated organization. Pay the required fees to the designated organization. Obtain a Letter of Support from the designated organization(s). Submit your application for permanent residency and work permit.The processing time for the Start-up Visa Program can vary, and the waiting period for a final decision on an application is typically between 18 months to 3 years. If you have any specific questions about the process, feel free to ask!"},
             {"role": "user", "content":"How to contact IQ Canada?"},
             {"role": "assistant", "content":"CALL US: +1 416 915 4230 | E-MAIL US: info@iqcanada.ca"},
             {"role": "user", "content": "How does IQ Canada assess the feasibility of my business idea for success in Canada?"},
             {"role": "assistant", "content": "IQ Canada conducts Concept Validation to evaluate the feasibility of your business idea, ensuring its potential for success in the Canadian market."},
             {"role": "user", "content": "What market insights does IQ Canada provide for my product or service in Canada?"},
             {"role": "assistant", "content": "With deep insights into the Canadian market, IQ Canada ensures your product or service meets the right audience and fulfills the market demand."},
             {"role": "user", "content": "How does IQ Canada help in penetrating the Canadian market effectively?"},
             {"role": "assistant", "content": "IQ Canada crafts bespoke Go-To-Market Strategies, allowing your business to penetrate the Canadian market effectively and efficiently."},
             {"role": "user", "content":prompt}
             ]+ch
         )
  return(response.choices[0].message.content)
def create_startup_visa_chatbot():
    conn = sqlite3.connect('startup_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS startup_qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_information (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            email TEXT,
            phone TEXT
        )
    ''')

    conn.commit()
    conn.close()
    conversation_history=[]
    def chatbot(user_input):
        nonlocal conversation_history
        user_input_processed = preprocess_text(user_input)
        conversation_history.append({"role": "user", "content": user_input})
        conn = sqlite3.connect("startup_database (2).db")
        cursor = conn.cursor()
        cursor.execute('SELECT question, answer FROM startup_qa')
        qa_data = cursor.fetchall()
        question_db = [preprocess_text(question) for question, _ in qa_data]
        answer_db = [answer for _, answer in qa_data]
        vectorizer = TfidfVectorizer()
        question_db.append(user_input_processed)
        question_vectors = vectorizer.fit_transform(question_db)
        cosine_similarities = cosine_similarity(question_vectors[-1], question_vectors[:-1]).flatten()
        most_similar_index = cosine_similarities.argmax()
        similarity_threshold = 0.5
        if cosine_similarities[most_similar_index] < similarity_threshold:
            response = gpt(user_input_processed, conversation_history)
        else:
            response = answer_db[most_similar_index]
        user_id = str(uuid.uuid4())
        pattern = r'NAME:\s*"([^"]+)"\s*EMAIL:\s*"([^"]+)"\s*PHONE:\s*"([^"]+)"'
        match = re.search(pattern, user_input)
        if match:
             name = match.group(1)
             email = match.group(2)
             phone = match.group(3)
             store_user_information(cursor,user_id,name,email,phone)
             conn.commit()
             conn.close()
        conversation_history.append({"role": "assistant", "content": response})

        return response

    return chatbot

def store_user_information(cursor, user_id, name,email,phone):
        conn = sqlite3.connect("startup_database (2).db")
        cursor.execute('''
            INSERT INTO user_information (user_id, name, email, phone) VALUES (?, ?, ?, ?)
        ''', (user_id, name, email, phone))
        cursor.execute('SELECT * FROM user_information')
        user_info_data = cursor.fetchall()
        conn.commit()
        conn.close()

chatbot_model=create_startup_visa_chatbot()
@app.route('/')
def index():
    return render_template("Frontend.html")


@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_history  # Access the global conversation_history variable
    user_input = request.form['user_input']
    
    if user_input.lower() in ['bye', 'exit', 'quit']:
        response = "Goodbye!"
        conversation_history = []  # Clear conversation history on user exit
    elif user_input.lower() in ['hi', 'hello', 'hey']:
        response = "Hello! How can I assist you today?"
    else:
        response = chatbot_model(user_input)

    return jsonify({'response': response})

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        return jsonify({'transcription': text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == "__main__":
    app.run(debug=False,port=3017)
