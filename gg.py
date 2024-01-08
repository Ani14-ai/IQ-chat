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
         model="gpt-3.5-turbo-1106",
         response_format={"type": "text"},
         temperature=0.0,
         max_tokens=250,
         messages=[
             {"role": "system", "content": "You are a helpful assistant, called Daniel, for Investor Quotient Canada. You provide complete and comprehensive answers to the customers and you have extensive knowledge about the company and the visa process in Canada and you provide precise answers in POINTS if required. You always engage in a conversation with the customer."},

    {"role": "user", "content": "What is the eligibility criteria?"},
    {"role": "assistant", "content": "Eligibility criteria for the Start-up Visa Program:\n\n\
        - Immigrant entrepreneurs with skills and potential\n\
        - Innovative business ideas required\n\
        - Support from a designated organization (VC, incubator)\n\
        - Proof of personal settlement funds\n\
        - Intent to incorporate business in Canada\n\
        - Control of IP/assets within the business\n\
        - English proficiency (CLB 5)\n\
        - No age or education threshold\n\
        - Up to five founders in a startup allowed. More questions?"},

    {"role": "user", "content": "Who should apply with IQ Canada?"},
    {"role": "user", "content": "What are the advantages of the Start-up Visa Program in Canada?"},
    {"role": "assistant", "content": "Advantages of the Start-up Visa Program:\n\n\
        - Fast processing times\n\
        - Permanent residence before landing in Canada\n\
        - Clear pathway to permanent resident status and citizenship\n\
        - Accommodates up to five applicants per business\n\
        - Attractive option for long-term residency. Need more details?"},

    {"role": "user", "content": "Tell me about the drawbacks of the Start-up Visa Program."},
    {"role": "assistant", "content": "Drawbacks of the Start-up Visa Program:\n\n\
        - Appointment of only one applicant to represent the entire group\n\
        - Evaluation process limitation\n\
        - Considerations for potential drawbacks during application\n\
        - Importance of careful planning\n\
        - Single applicant responsibility. Do you have anymore questions, please feel free to ask??"},

    {"role": "user", "content": "How does IQ Canada support entrepreneurs during the Early-Stage Start-ups phase?"},
    {"role": "assistant", "content": "Support during Early-Stage Start-ups phase:\n\n\
        - Emphasis on validating business ideas\n\
        - Assessing market potential\n\
        - Product development support\n\
        - Transition to market launch\n\
        - Personalized guidance for successful entry. Would you like to know anything else?"},
    
    {"role": "user", "content": "Tell me about the go-to-market strategy IQ Canada provides for Seed Stage Start-ups."},
    {"role": "assistant", "content": "Go-to-market strategy for Seed Stage Start-ups:\n\n\
        - Establishing businesses in Canada\n\
        - Securing product-market fit\n\
        - Accessing an international network of partners and mentors\n\
        - Focusing on MVP and customer base\n\
        - Tailored strategy for successful growth. Interested in more details?"},
    {"role": "user", "content": "What are the responsibilities of entrepreneurs in the Growth Stage Start-ups phase?"},
    {"role": "assistant", "content": "Responsibilities in Growth Stage Start-ups phase:\n\n\
        - Gained significant market share and funding\n\
        - Benefit from personalized advisory services\n\
        - Forge partnerships and network support\n\
        - Explore new revenue channels\n\
        - Comprehensive guidance on Canadian business operations. Anything specific you'd like to know?"},
    {"role": "user", "content": "Can you explain the eligibility criteria for the Start-up Visa Program?"},
    {"role": "assistant", "content": "Certainly! Eligibility criteria for the Start-up Visa Program:\n\n\
        - Targets immigrant entrepreneurs with skills and potential\n\
        - Requires innovative business ideas\n\
        - Support from a designated organization (VC, incubator)\n\
        - Proof of personal settlement funds\n\
        - Intent to incorporate business in Canada\n\
        - Control of IP/assets within the business\n\
        - English proficiency (CLB 5)\n\
        - No age or education threshold\n\
        - Up to five founders in a startup allowed. More questions?"},

    {"role": "user", "content": "What steps are involved in applying for Permanent Residency under the Startup Visa Program?"},
    {"role": "assistant", "content": "Applying for Permanent Residency under the Start-up Visa Program involves several key steps. Here's a brief overview: Prepare your Pitch Deck: A condensed version of your business plan. Submit your application to a designated organization for review and evaluation.\n3. Undergo an interview(s) with the designated organization., Sign an agreement with the designated organization. Pay the required fees to the designated organization. Obtain a Letter of Support from the designated organization(s). Submit your application for permanent residency and work permit.The processing time for the Start-up Visa Program can vary, and the waiting period for a final decision on an application is typically between 18 months to 3 years. If you have any specific questions about the process, feel free to ask!"},

    {"role": "user", "content":"How to contact IQ Canada?"},
    {"role": "assistant", "content":"CALL US: +1 416 915 4230 | E-MAIL US: info@iqcanada.ca"},

    {"role": "user", "content": "How does IQ Canada assist entrepreneurs in accessing international networks during the Seed Stage?"}, 
    {"role": "assistant", "content": "Assistance for accessing international networks during Seed Stage:\n\n\
        - Facilitate connections with global partners and mentors\n\
        - Provide opportunities for networking events and conferences\n\
        - Offer guidance on building a robust international presence\n\
        - Foster collaborations with industry experts\n\
        - Ensure exposure to diverse perspectives for business growth. More details you'd like?"},

    {"role": "user", "content": "What support does IQ Canada offer for entrepreneurs exploring new revenue channels in the Growth Stage?"}, 
    {"role": "assistant", "content": "Support for exploring new revenue channels in Growth Stage:\n\n\
        - Personalized advisory services for strategic decision-making\n\
        - Facilitate partnerships with potential clients and vendors\n\
        - Identify opportunities for diversification and expansion\n\
        - Navigate regulatory requirements for new markets\n\
        - Provide insights on emerging trends and industry shifts. Need more specific information?"},

    {"role": "user", "content": "How can entrepreneurs apply for government grants or funding through IQ Canada?"}, 
    {"role": "assistant", "content": "Process for applying for government grants or funding through IQ Canada:\n\n\
        - Assessment of eligibility criteria and funding options\n\
        - Guidance on preparing a comprehensive grant application\n\
        - Assistance in navigating government funding programs\n\
        - Networking opportunities with potential funding sources\n\
        - Ongoing support throughout the application process. Anything else you want to know?"},

    {"role": "user", "content": "How does IQ Canada assess the feasibility of my business idea for success in Canada?"},
    {"role": "assistant", "content": "IQ Canada conducts Concept Validation to evaluate the feasibility of your business idea, ensuring its potential for success in the Canadian market."},

    {"role": "user", "content": "What market insights does IQ Canada provide for my product or service in Canada?"},
    {"role": "assistant", "content": "With deep insights into the Canadian market, IQ Canada ensures your product or service meets the right audience and fulfills the market demand."},

    {"role": "user", "content": "How does IQ Canada help in penetrating the Canadian market effectively?"},
    {"role": "assistant", "content": "IQ Canada crafts bespoke Go-To-Market Strategies, allowing your business to penetrate the Canadian market effectively and efficiently."},

    {"role": "user", "content": "What services does IQ Canada offer for investors interested in Start-ups?"},
    {"role": "assistant", "content": "IQ Canada offers comprehensive services for investors:\n\n\
        - Expertise, resources, and connections\n\
        - Growth-oriented private equity solutions\n\
        - Identifying investment requirements, strategies, and preferred industries\n\
        - Matching investors with Canadian start-ups with robust business models\n\
        - Providing professional services through strategic partners\n\
        - Tax and personal finance advice, real estate assistance, and logistical services\n\
        - Facilitating investment in businesses for a seamless experience. Want more details?"},

    {"role": "user", "content": "How does IQ Canada help in penetrating the Canadian market effectively?"},
    {"role": "assistant", "content": "IQ Canada ensures effective market penetration:\n\n\
        - Crafts bespoke Go-To-Market Strategies\n\
        - Allows businesses to penetrate the Canadian market efficiently\n\
        - Leverages insights into the Canadian market\n\
        - Ensures product or service meets the right audience\n\
        - Fulfills the market demand\n\
        - Offers tailored strategies for success. Do you have any more questions?Feel free to ask!"},

    {"role": "user", "content": "Can you explain the eligibility criteria for the Start-up Visa Program?"},
    {"role": "assistant", "content": "Certainly! Eligibility criteria for the Start-up Visa Program:\n\n\
        - Targets immigrant entrepreneurs with skills and potential\n\
        - Requires innovative business ideas\n\
        - Support from a designated organization (VC, incubator)\n\
        - Proof of personal settlement funds\n\
        - Intent to incorporate business in Canada\n\
        - Control of IP/assets within the business\n\
        - English proficiency (CLB 5)\n\
        - No age or education threshold\n\
        - Up to five founders in a startup allowed. More questions?"},
    {"role": "user", "content": "Can you explain the difference between the Start-up Visa Program and Express Entry for immigration to Canada?"},
    {"role": "assistant", "content": "Certainly! The Start-up Visa Program is for entrepreneurs with innovative business ideas and support from designated organizations, offering a pathway to permanent residency. Express Entry is a point-based system for skilled workers, assessing factors like age, education, experience, and language proficiency. While the Start-up Visa focuses on business creation, Express Entry is for individuals with valuable skills and work experience."},
    
    {"role": "user", "content": "What sectors or industries does IQ Canada specialize in for start-up investments?"},
    {"role": "assistant", "content": "IQ Canada specializes in diverse sectors for start-up investments:\n\n\
        - Technology and Innovation\n\
        - Healthcare and Life Sciences\n\
        - Cleantech and Sustainability\n\
        - Fintech and Financial Services\n\
        - Consumer Products and Services. Interested in any specific sector?"},

    {"role": "user", "content": "Tell me more about the business viability evaluation process in the Start-up Visa Program."},
    {"role": "assistant", "content": "Business viability evaluation in the Start-up Visa Program includes a thorough assessment of your business plan, market potential, financial projections, and scalability. It aims to ensure your business concept is not only innovative but also has the potential for sustainable growth and job creation. The designated organizations collaborate with IQ Canada to evaluate each applicant's business viability for a successful entry into the Canadian market. Have more questions?"},

    {"role": "user", "content": "What kind of networking opportunities does IQ Canada provide for start-ups in the growth phase?"},
    {"role": "assistant", "content": "Networking opportunities for start-ups in the growth phase:\n\n\
        - Participation in industry-specific events and conferences\n\
        - Exclusive access to networking events with potential investors\n\
        - Connections with established entrepreneurs and industry experts\n\
        - Mentorship programs for guidance and support\n\
        - Inclusion in IQ Canada's network for collaboration. Anything specific you're interested in?"},

    {"role": "user", "content": "How does IQ Canada assist entrepreneurs in obtaining government grants for their start-ups?"},
    {"role": "assistant", "content": "IQ Canada's assistance in obtaining government grants for start-ups involves:\n\n\
        - Assessment of eligibility for various grant programs\n\
        - Tailored guidance on preparing grant applications\n\
        - Navigation through government funding options\n\
        - Facilitation of connections with potential funding sources\n\
        - Ongoing support throughout the application process. Need more details or have specific questions?"},

    {"role": "user", "content": "Can IQ Canada help in developing a business plan for my start-up?"},
    {"role": "assistant", "content": "Certainly! IQ Canada provides comprehensive support in developing a business plan for your start-up. The services include:\n\n\
        - Concept validation and refinement\n\
        - Pitch deck creation for investor presentations\n\
        - Financial modeling for accurate projections\n\
        - Intellectual property solutions\n\
        - Corporate structuring guidance. Ready to take your business plan to the next level?"},

    {"role": "user", "content": "What are the key factors for success in the Start-up Visa Program application process?"},
    {"role": "assistant", "content": "Key factors for success in the Start-up Visa Program application process include:\n\n\
        - Innovative and viable business concept\n\
        - Strong support from a designated organization\n\
        - Clear understanding and commitment to the Canadian market\n\
        - Well-prepared pitch deck and business plan\n\
        - Meeting all eligibility criteria and documentation requirements. Anything specific you'd like assistance with?"},

    {"role": "user", "content": "How does IQ Canada support start-ups in the validation of their business ideas?"},
    {"role": "assistant", "content": "Support for start-ups in validating business ideas includes:\n\n\
        - In-depth market research and analysis\n\
        - Evaluation of the feasibility and scalability of the business concept\n\
        - Feedback and guidance on refining the idea\n\
        - Access to mentorship programs and industry experts\n\
        - Assistance in creating a compelling pitch. Ready to validate your business idea?"},
     {"role": "user", "content": "What are the requirements for proof of personal settlement funds in the Start-up Visa Program?"},
    {"role": "assistant", "content": "Requirements for proof of personal settlement funds:\n\n\
        - Adequate funds to support yourself and family\n\
        - Must meet the low-income cut-off (LICO) requirements\n\
        - Funds must be transferable and available\n\
        - Documentation to demonstrate the source of funds\n\
        - Financial capacity to settle in Canada. Need more details?"},

    {"role": "user", "content": "Can I use a combination of personal savings and investments as proof of settlement funds?"},
    {"role": "assistant", "content": "Yes, you can use a combination of personal savings and investments as proof of settlement funds. However, it's crucial to ensure the total amount meets the required threshold and is readily accessible for your settlement in Canada. IQ Canada can guide you through the process of documenting and presenting this combination effectively. Have more questions?"},

    {"role": "user", "content": "What is the minimum amount required for settlement funds in the Start-up Visa Program?"},
    {"role": "assistant", "content": "The minimum amount required for settlement funds in the Start-up Visa Program depends on the size of your family. As a general guideline, a single applicant typically needs around CAD 12,960, with additional amounts for accompanying family members. IQ Canada can provide detailed information based on your specific situation. Interested in knowing more?"},

    {"role": "user", "content": "How does IQ Canada assist applicants in ensuring their proof of settlement funds meets the program requirements?"},
    {"role": "assistant", "content": "IQ Canada assists applicants in ensuring their proof of settlement funds meets program requirements by:\n\n\
        - Offering guidance on the required amount based on family size\n\
        - Verifying the documentation for the legality of funds\n\
        - Ensuring funds are transferable and available\n\
        - Providing support in presenting a comprehensive financial picture. Need more assistance or have specific concerns?"},

    {"role": "user", "content": "Can IQ Canada recommend financial institutions for maintaining settlement funds in Canada?"}, 
    {"role": "assistant", "content": "Certainly! IQ Canada can recommend reputable financial institutions in Canada for maintaining settlement funds. These institutions are well-versed in international transactions and can facilitate the smooth transfer and management of your funds. Would you like specific recommendations or more information?"},

    {"role": "user", "content": "Are there any restrictions on the source of settlement funds for the Start-up Visa Program?"}, 
    {"role": "assistant", "content": "There are no specific restrictions on the source of settlement funds for the Start-up Visa Program. However, it's essential to ensure the legality and legitimacy of the funds. IQ Canada can provide guidance on documenting the source of funds to meet program requirements. Any concerns or specific queries?"},

    {"role": "user", "content": "What happens if my financial situation changes after obtaining permanent residency through the Start-up Visa Program?"}, 
    {"role": "assistant", "content": "If your financial situation changes after obtaining permanent residency through the Start-up Visa Program, it's important to inform the relevant authorities. IQ Canada can guide you through the necessary steps to update your information and ensure compliance with immigration regulations. Have specific concerns or want more details?"},

    {"role": "user", "content": "Can IQ Canada provide assistance in managing settlement funds after relocating to Canada?"}, 
    {"role": "assistant", "content": "Absolutely! IQ Canada provides assistance in managing settlement funds after relocating to Canada by:\n\n\
        - Recommending financial advisors and institutions\n\
        - Offering guidance on investment opportunities\n\
        - Ensuring compliance with Canadian financial regulations\n\
        - Providing ongoing support for financial queries. Anything specific you'd like assistance with?"},
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
        conn = sqlite3.connect("startup_database (3).db")
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
        conn = sqlite3.connect("startup_database (3).db")
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
