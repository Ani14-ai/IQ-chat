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
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    return ' '.join(tokens)

def gpt(prompt,ch):
  response= client.chat.completions.create(
         model="gpt-4",
         response_format={"type": "text"},
         temperature=0.1,
         max_tokens=100,
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
    startup_questions = [
    "What is the current market scenario/key trends in the market?",
	"What is the contact/email for IQ Canada?",
	"Where is IQ Canada's office?",
	"Template for Completion Checklist?",
	"How do I find out if my business is innovative?",
	"What are the fees charged by Designated Organizations?",
	"How do you help to launch my business in Canada?",
	"Who are some of your clients?",
	"What is your success rate?",
	"Contact for Proof of Settlements of Funds?",
	"Does IQ Canada have experience or expertise in supporting startups within the Oil & Gas and Energy sector?",
	"How can IQ Canada help ensure that my startup complies with all the relevant regulations and requirements in the industry in Canada?",
	"Can IQ Canada provide insights or support in conducting market research for the specified industry to ensure a strong understanding of the competitive landscape and potential market challenges?",
	"Does IQ Canada have knowledge or partnerships with technology providers, to support the development of the platform?",
	"How can IQ Canada assist in refining and optimizing the business model considering the unique challenges and opportunities in the industry?",
	"What legal services does IQ Canada offer to ensure the proper setup of contracts, agreements, and partnerships, especially in a complex industry?"]
    startup_answers = [
	"The Canadian startup ecosystem is thriving, marked by a surge in innovative companies across sectors. The government actively supports startups through funding, accelerators, and tax incentives, fostering a favorable environment that attracts global talent and investment. Key trends include the transformative impact of Artificial Intelligence (AI) and Machine Learning (ML), the rising value of data analytics, disruptive advancements in Fintech, technological innovations in Health Tech for improved healthcare, and the significant role startups play in addressing environmental challenges through sustainability solutions. IQ Canada is dedicated to aiding Canadian startups in navigating this dynamic market, emphasizing innovation and entrepreneurship as drivers for economic growth and social progress.",
	"CALL US: +1 416 915 4230 | E-MAIL US: info@iqcanada.ca",
	"ADDRESS: 100 King St W Suite 5700, Toronto, ON M5X 1C7, Canada",
	"A template checklist outlines the typical steps involved in completing a series A investment transaction, starting from signing to completion. The items are examples, and individuals creating their checklist should refer to key transaction documents, including a subscription agreement, shareholders' agreement, and a new constitution. The checklist is not exhaustive and excludes certain complexities like a rolling close, investment tranches, or conversion of debt or convertible notes. Reach out to IQ Canada's support team for additional information on the completion checklist.",
	"To determine if a business is innovative, consider factors such as uniqueness, potential market disruption, use of new technology, intellectual property protection, and scalability. If the business meets one or more of these criteria, it may be considered innovative. However, the final determination rests with the designated organization providing a Letter of Support.",
	"Fees can range between $25,000 and $50,000. TiE Toronto fees are $25,000.",
	"Launching a business in Canada with IQ Canada involves a systematic six-phase approach. In the first phase, they provide startup visa advisory services, transforming clients' ideas into viable business propositions, offering market intelligence, creating plans, and assisting in the visa application process. The second phase focuses on developing a go-to-market plan, defining target markets, engaging with stakeholders, and establishing a digital presence. In the third phase, IQ Canada helps companies build Minimum Viable Products (MVPs) and prototypes with a strong emphasis on UI/UX design and quality assurance. The fourth phase involves identifying and screening investors, coaching clients for presentations, and managing fundraising due diligence. In the fifth phase, IQ Canada guides clients through the startup visa designated entry process, leveraging their expertise in the Canadian startup ecosystem. The final phase includes incorporating the Canadian business and protecting intellectual property through collaboration with leading law firms, facilitating the entire incorporation process, and assisting in filing patents, trademarks, and other IP protection applications.",
	"Some of our clients include companies such as: Lawkin.ca, Farms.ca, iSpeed.ca, gotchosen.ca, hiringplug.com, onecircle.ca, CognitoCRM Vietnam, Vite, D2P Global Vietnam",
	"Our Stellar Track Record of Success includes 250+ successful start-ups; 550+ founders supported; 40+ mentors and advisors onboard.",
	"The letter should be printed on the bank’s letterhead with your contact information included. You should also provide us with your original signature at the bottom. Should you have any questions, please contact Upasana Sharma at usharma@iqcanada.ca",
	"IQ Canada indeed has experience in supporting startups within the Oil & Gas and Energy sector. It has worked on projects which are specifically designed to address the challenges in this industry. While specific information cannot be disclosed, this platform is tailored for businesses in Oil & Gas, Agriculture, Power Plants, Waste Water Plants, Food Processing, and the Chemical Industry.",
	"IQ Canada can assist your startup in ensuring compliance with relevant regulations in the industry in Canada. The platform development takes into account the regulatory landscape, and IQ Canada's expertise extends to guiding startups through compliance requirements.",
	"Based on the projects IQ Canada has worked on which has advanced features, such as data analytics and real-time monitoring, IQ Canada can likely provide support or insights in conducting comprehensive market research for your specific industry needs.",
	"IQ Canada showcases a commitment to using the latest technologies for web-based applications. It has recently worked on a project based on the LAMP stack. This choice indicates a familiarity and potential expertise in the LAMP stack, ensuring robust development capabilities. It also has collaborations to choose the latest technology that will align with your business model."
	"IQ Canada can play a pivotal role in optimizing the business model for your business. It can include features like offering predictive analytics,etc indicating a strategic approach tailored for the complexities of your industry.",
	"IQ Canada has partnerships or expertise in the legal aspects relevant to your industry. This includes setting up contracts, agreements, and navigating the legal intricacies in this sector."]
    
    question_db = [preprocess_text(question) for question in startup_questions]
    answer_db = startup_answers
    conversation_history = []

    def chatbot(user_input):
        nonlocal conversation_history, question_db

        user_input_processed = preprocess_text(user_input)
        conversation_history.append({"role": "user", "content": user_input_processed})
        vectorizer = TfidfVectorizer()
        question_db.append(user_input_processed)
        question_vectors = vectorizer.fit_transform(question_db)        
        cosine_similarities = cosine_similarity(question_vectors[-1], question_vectors[:-1]).flatten()
        most_similar_index = cosine_similarities.argmax()
        similarity_threshold = 0.5
        if most_similar_index < len(answer_db) and cosine_similarities[most_similar_index] >= similarity_threshold:
            response = answer_db[most_similar_index]
        else:
             response = gpt(user_input_processed, conversation_history)

        conversation_history.append({"role": "assistant", "content": response})
        return response

    return chatbot

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
