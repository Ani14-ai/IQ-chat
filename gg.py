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
load_dotenv()
secret_key = os.getenv("SECRET_KEY")
client = OpenAI(api_key=os.getenv("openai_key"))
app = Flask(__name__)
socketio = SocketIO(app, port=3012, cors_allowed_origins="*")
CORS(app, resources={"/api/*": {"origins": "*"}})
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    return ' '.join(tokens)

def query_gpt4_api(question, dest_language='en'):
     response = client.chat.completions.create(
         model="gpt-3.5-turbo-1106",
         response_format={"type": "text"},
         max_tokens=50,
         messages=[
             {"role": "system", "content": "You are a helpful assistant for Investor quotient Canada. Your job is to answer any Query related to Startup Visa in Canada and you provide a to the point answer in paragraphs"},
             {"role": "user", "content": question}
         ]
     )
     return response.choices[0].message.content
 
def create_chatbot(question_db, answer_db):
    def preprocess_database(database):
        return [preprocess_text(question) for question in database]
    
    question_db_processed = preprocess_database(question_db)

    def chatbot(user_input):
        nonlocal question_db, question_db_processed, answer_db

        # Preprocess user input
        user_input_processed = preprocess_text(user_input)

        # Vectorize the questions
        vectorizer = TfidfVectorizer()
        question_vectors = vectorizer.fit_transform(question_db_processed)

        # Calculate cosine similarity between the user input and existing questions
        user_vector = vectorizer.transform([user_input_processed])
        cosine_similarities = cosine_similarity(user_vector, question_vectors).flatten()
        most_similar_index = cosine_similarities.argmax()
        similarity_threshold = 0.5

        if cosine_similarities[most_similar_index] < similarity_threshold:
            response = query_gpt4_api(user_input_processed)
        else:
            response = answer_db[most_similar_index]
        question_db_processed.append(user_input_processed)

        # Return the corresponding answer
        response = answer_db[most_similar_index]

        return response

    return chatbot  # Return the chatbot function

def create_startup_visa_chatbot():
    startup_questions = [
       "What is the eligibility criteria for the startup visa program (SUV)?",
       "How many co-founders are allowed to apply for a startup visa together?",
       "Can I apply for a startup visa if I don't meet the age or education threshold?",
       "What are the language requirements for the startup visa program?",
       "How do I qualify for applying for a startup visa?",
       "What is a Letter of Support in the context of the startup visa program?",
       "How does the startup visa program support businesses in Canada?",
       "What are the steps required to apply for Permanent Residency under the startup visa program?",
       "What is the processing time for the startup visa program?",
       "Why should you think of coming to Canada under the Startup Visa Program?",
       "Why is Ontario considered a preferred destination for startups?",
       "Can I apply for a startup visa if my business is not yet incorporated in Canada?",
       "What are the fees charged by designated organizations for the startup visa program?",
       "How can I find out if my business is considered innovative for the startup visa program?",
       "Are there any government programs that support new startups in Canada?",
       "What are the requirements for proof of personal settlement funds when applying for a startup visa?",
       "Can I include family members in my startup visa application, and what are the requirements for them?",
       "How can I secure a Letter of Support from a designated organization for my startup visa application?",
       "Are there restrictions on the type of business that qualifies for a startup visa?",
       "What is the role of venture capital funds, angel investor groups, and business incubators as designated organizations?",
       "Is there a specific minimum investment requirement from designated organizations to qualify for a startup visa?",
       "How long does it take to get a response after applying to a designated organization?",
       "What are the benefits of obtaining Canadian Permanent Residency through the startup visa program?",
       "Can I apply for a startup visa if I already have a business established in another country?",
       "How does the Global Talent Stream in Ontario facilitate access to tech talent for entrepreneurs?",
       "Are there specific industries or sectors that the startup visa program prioritizes?",
       "What role does innovation play in the assessment of a startup's eligibility for the visa program?",
       "How does the startup visa program contribute to job creation in Canada?",
       "Are there any restrictions on the nationality of founders applying for a startup visa?",
       "Can a startup with a focus on sustainable or social impact qualify for the visa program?",
       "How does the startup visa program evaluate the innovative nature of a business idea?",
       "Can a startup visa holder bring existing employees from their home country to work in the Canadian business?",
       "What role does the business plan play in the startup visa application process?",
       "Are there specific requirements for demonstrating the potential to create jobs for Canadians in the startup visa application?",
       "How does the startup visa program handle partnerships and collaborations with other businesses?",
       "Can an entrepreneur who previously had a startup in another country apply for a startup visa in Canada?",
       "What support does the Canadian government offer to startups beyond the startup visa program?",
       "Are there any specific requirements for startups in the technology sector applying for a visa?",
       "How does the startup visa program assess the market viability of a business idea?",
       "Can a startup visa holder sponsor family members for Canadian Permanent Residency?",
       "What happens if a designated organization withdraws its support after providing a Letter of Support?",
       "How does the startup visa program contribute to the growth of the Canadian tech industry?",
       "Can a startup visa holder apply for additional permits for employees hired in Canada?",
       "What role do language proficiency requirements play in the startup visa application?",
       "How does the Canadian government ensure compliance with the job creation commitments made by startup visa holders?",
       "Can a startup visa holder apply for Canadian citizenship immediately upon obtaining Permanent Residency?",
       "What documentation is required to demonstrate control of intellectual property/assets within the business?",
       "How does the startup visa program facilitate networking and collaboration among entrepreneurs?",
       "Are there any specific criteria for measuring the global competitiveness of a startup under the program?",
       "Can a startup visa holder apply for permanent residency if their business fails to meet the specified criteria?",
       "What is a Designated Organization?",
       "How does IQ Canada help startup businesses?",
       "What are the instructions for obtaining Proof of Settlement Funds Letter from the bank",
       "What are the funds required in Canadian dollars for family members?",
       "What is the minimum necessary income required according to size of family unit?",
       "What is the application process for the Start-up Visa Program?",
       "What is the difference between support from venture capital funds, angel investor groups, and business incubators?",
       "Who will review my Start-up Visa Program application?",
       "What is an essential person?",
       "Why is there a different minimum investment from a venture capital fund and an angel investor group?",
       "Do I have to invest my own money if I want to apply through the Start-up Visa Program?",
       "What happens if I receive investment support from more than one designated organization?",
       "How much money is required for Canada startup visa?",
       "How to apply for a startup visa?",
       "What is the success rate of startup visa in Canada?",
       "List of Designated Organizations/Venture Capital funds/Angel Investor Groups/Business incubators?",
       "What specific qualifications or experience should co-founders possess to enhance the chances of a successful startup visa application?",
       "Can a startup with a physical presence outside of Canada still be eligible for the Startup Visa Program?",
       "What happens if there are changes in the ownership structure or leadership team after receiving the startup visa?",
       "Can a startup that has previously received funding or support from non-designated organizations still apply for the visa program?",
       "Are there specific requirements for the business plan, and what key elements should it include to strengthen the application?",
       "How does the Startup Visa Program handle intellectual property rights and protection for innovative business ideas?",
       "Can a startup visa holder switch to a different business idea or industry after obtaining Permanent Residency?",
       "What role does the province or territory play in the Startup Visa Program, and are there specific advantages to choosing a particular location?",
       "Can an entrepreneur who has previously applied for other Canadian immigration programs still apply for the Startup Visa Program?",
       "How does the Startup Visa Program contribute to diversity and inclusion in the Canadian entrepreneurial ecosystem?",
       "Can a startup visa holder pursue additional education or training while running their business in Canada?",
       "What support is available for startups in the post-approval phase of the visa program?",
       "Are there any tax incentives or benefits for startups that go through the Startup Visa Program?",
       "How does the program handle situations where a designated organization goes out of business or is no longer able to provide support?",
       "Can a startup visa holder sell their business and still maintain their Permanent Residency status in Canada?",
       "Are there specific criteria or benchmarks that a startup must meet to demonstrate its potential for scalability and growth?",
       "What role does the startup ecosystem in Canada play in supporting and integrating startup visa holders?",
       "How does the Startup Visa Program accommodate businesses in emerging or unconventional industries?",
       "Are there any restrictions on the use of funds obtained through designated organizations for the startup?",
       "Can a startup visa holder bring in foreign partners or collaborators to contribute to the business?",
       "What steps should an entrepreneur take if they experience challenges or setbacks in meeting the job creation targets outlined in their business plan?",
       "How does the program handle situations where the business pivot is necessary for the startup's survival or success?",
       "Can a startup visa holder apply for additional funding or support from designated organizations after the initial approval?",
       "Are there any specific tax obligations or considerations for startup visa holders operating in Canada?",
       "How does the program address concerns related to the competitive landscape and market saturation in specific industries?",
       "Can a startup visa holder participate in other entrepreneurial activities or ventures outside of their designated business?",
       "What happens if a designated organization provides a Letter of Support but later disagrees with the direction or management of the startup?",
       "Are there specific requirements for startups in the healthcare or biotechnology sectors applying for the visa program?",
       "Can a startup visa holder hire foreign workers to contribute to the growth of their business in Canada?",
       "How does the program handle situations where a startup is acquired by another company?",
       "Are there any restrictions on the types of investments that can be made by designated organizations in startups?",
       "How does the program address concerns related to the sustainability and long-term viability of a startup?",
       "Can a startup visa holder apply for additional permits for international employees who are crucial to the business's success?",
       "What resources are available to help startup visa holders integrate into the Canadian business community and culture?",
       "How does the Startup Visa Program handle partnerships with research institutions or universities for innovative startups?",
       "Can a startup visa holder relocate their business to a different province or territory within Canada after obtaining Permanent Residency?",
       "What role does the startup community play in supporting newcomers through networking events and collaborative initiatives?",
       "Are there any specific measures in place to protect against fraud or misrepresentation in the startup visa application process?",
       "How does the program support startups in rural or remote areas of Canada, away from major urban centers?",
       "Can a startup visa holder take a temporary leave from their business without jeopardizing their immigration status?",
       "What avenues are available for dispute resolution between co-founders or with designated organizations during the application process?",
       "How does the program account for external factors such as economic downturns or market fluctuations that may impact startups?",
       "Can a startup visa holder sponsor skilled international workers through other immigration programs?",
       "How does the program encourage collaboration and knowledge-sharing among startups within the same industry or sector?",
       "Are there specific criteria for demonstrating the potential for international market expansion for a startup?",
       "Can a startup visa holder access government grants or funding programs to further support their business growth?",
       "How does the program address concerns related to the protection of sensitive business information and trade secrets?",
       "Can a startup visa holder participate in accelerator programs or business competitions to further develop their business?",
       "What role does the Startup Visa Program play in fostering innovation in traditional industries or sectors?",
       "Can a startup visa holder apply for a work permit for a key employee who is not eligible for permanent residency?",
       "How does the program assess the impact of a startup on the local community and economy?",
       "Are there specific initiatives or resources to support startups led by underrepresented groups, such as women or minorities?",
       "Can a startup visa holder apply for additional funding from other sources after the initial investment from a designated organization?",
       "How does the program handle situations where a startup pivots its business model or focus after receiving approval?",
       "What are the documents required for application/ document checklist?"
   ] 
    startup_answers = [
       "To be eligible for Canada's Start-up Visa Program, entrepreneurs must have the skills and potential to build businesses in Canada that are innovative, can create jobs for Canadians, and can compete on a global scale. Additionally, they must have the support of one of the designated organizations, such as a venture capital fund or business incubator, and meet other requirements such as proof of personal settlement funds, intent to incorporate a business in Canada, control of IP/assets within the business, and English proficiency (CLB 5). There is no age or education threshold, and up to 5 founders in a startup are allowed.",
       "Up to 5 founders in a startup are allowed to apply for Canada's Start-up Visa Program together. Each applicant must hold at least 10% of the voting rights in the business, and all applicants combined must hold over 50% of the voting rights.",
       "The startup visa program does not have age or education thresholds, providing flexibility for a diverse range of entrepreneurs.",
       "Language requirements include meeting a minimum CLB 5 in all four language abilities: reading, writing, speaking, and listening.",
       "To qualify for the Start-up Visa Program, you must meet eligibility criteria, secure a Letter of Support from a designated organization, meet language requirements, have sufficient settlement funds, and pass security and medical clearances.",
       "In the context of the Start-up Visa Program, a Letter of Support is a document provided by a designated organization that supports the entrepreneur's business idea and confirms that the organization is investing in or supporting the business.",
       "Canada's Start-up Visa Program supports innovative businesses in Canada by providing access to a talented pool of individuals, government support for innovation, and a direct pathway to immigration in Canada.",
       "The steps include preparing a pitch deck, submitting the application to a designated organization, undergoing interviews, signing an agreement, paying required fees, obtaining a Letter of Support, and submitting the application for permanent residency and work permit.",
       "The processing time for the startup visa program can vary, with a general waiting period of 18 months to 3 years for the final decision on an application.",
       "The Startup Visa program is intended to attract international entrepreneurs who wish to establish new, scalable businesses in Canada which will support innovation and job creation.",
       "Ontario is a preferred destination for entrepreneurs due to its strategic location, economic strength, AI innovation hub, and quick access to tech talent through the Global Talent Stream.",
       "Yes, you can apply for a startup visa even if your business is not yet incorporated in Canada.",
       "Fees charged by designated organizations can range between $25,000 and $50,000, with specific amounts for organizations like TiE Toronto.",
       "To determine if your business is innovative, consider factors like uniqueness, market disruption potential, technology use, intellectual property, and scalability.",
       "Yes, there are over 150 government programs that support new and innovative startups in Canada, including SR&ED and iRAP.",
       "Proof of personal settlement funds is required to demonstrate that you have the financial means to support yourself and your family members when you arrive in Canada.",
       "Family members can be included in the startup visa application, and requirements for them include meeting security and medical clearances.",
       "To secure a Letter of Support, demonstrate strong market validation and scalability. Work with specialized business consultants for idea development and validation.",
       "The startup visa program does not impose restrictions on the type of business, but it must be innovative, have job creation potential, and compete globally.",
       "Designated organizations, such as venture capital funds, angel investor groups, and business incubators, play a crucial role by providing a Letter of Support to endorse and support the entrepreneur's business idea.",
       "While there's no specific minimum investment requirement, the financial support provided by designated organizations can vary.",
       "Response times from designated organizations can vary, and it's advisable to check with each organization individually for their specific timelines.",
       "Benefits of obtaining Canadian Permanent Residency include the ability to live and work in Canada, access to government programs supporting startups, and opportunities for business growth.",
       "Yes, you can apply for a startup visa even if you have a business established in another country. The startup must, however, be incorporated in Canada.",
       "The Global Talent Stream in Ontario allows employers to receive Labor Market Impact Assessments in ten business days, facilitating faster access to tech talent.",
       "While the startup visa program is open to various industries, Ontario's focus on AI and its large talent pool makes it particularly attractive for tech startups.",
       "Innovation is a key factor in the assessment of a startup's eligibility. The business should offer a unique product or service, disrupt the market, use emerging technology, have intellectual property protection, and demonstrate scalability.",
       "The startup visa program contributes to job creation by attracting innovative entrepreneurs who establish and grow businesses, leading to increased employment opportunities.",
       "There are no restrictions on the nationality of founders applying for a startup visa. The program is open to international entrepreneurs.",
       "Yes, a startup with a focus on sustainable or social impact can qualify for the visa program. Innovation in addressing social or environmental challenges is highly valued.",
       "The evaluation of the innovative nature of a business idea includes factors like uniqueness, market disruption potential, use of emerging technology, intellectual property, and scalability.",
       "Yes, a startup visa holder can bring existing employees to work in the Canadian business, contributing to the job creation aspect.",
       "The business plan is a crucial component of the startup visa application process, outlining the vision, mission, market strategy, and financial projections.",
       "Specific requirements for demonstrating job creation potential often involve detailing the roles and responsibilities of prospective hires, contributing to the Canadian job market.",
       "Partnerships and collaborations are viewed positively, showcasing the potential for networking and industry integration.",
       "Yes, an entrepreneur with previous startup experience in another country can apply for a startup visa in Canada.",
       "Beyond the startup visa program, the Canadian government offers various support programs, grants, and resources to foster startup growth.",
       "While there are no specific requirements for sectors, technology startups are encouraged, given their potential for innovation and global competitiveness.",
       "Market viability is assessed by examining market research, target audience analysis, and the potential demand for the product or service.",
       "Yes, a startup visa holder can sponsor eligible family members for Canadian Permanent Residency.",
       "If a designated organization withdraws support after providing a Letter of Support, the entrepreneur may need to seek support from an alternative designated organization.",
       "The startup visa program contributes to the growth of the Canadian tech industry by attracting and supporting innovative tech startups.",
       "Startup visa holders can apply for additional permits, such as work permits, for employees hired in Canada.",
       "Language proficiency requirements, typically at CLB 5, ensure effective communication and integration into the Canadian business environment.",
       "Compliance with job creation commitments is monitored, and startups failing to meet criteria may face consequences, including potential visa revocation.",
       "While obtaining Permanent Residency is a step toward citizenship, there are specific residency and time requirements for Canadian citizenship.",
       "Documentation demonstrating control of intellectual property/assets includes patents, trademarks, or copyrights protecting the startup's product or service.",
       "The startup visa program facilitates networking and collaboration through events, mentorship programs, and connections with industry experts.",
       "Global competitiveness is measured by assessing the startup's potential to compete on an international scale.",
       "A startup visa holder can apply for permanent residency, even if the business faces challenges, but individual cases may vary.",
       "The Start-up Visa Program requires applicants to secure a Letter of Support from a designated organization. Designated organizations are business groups that have been approved by the Canadian government to invest in or support start-ups through the Start-up Visa Program. These organizations include venture capital funds, angel investor groups, and business incubators.",
       "IQ Canada helps startup businesses through concept validation, market research, go-to-market strategies, comprehensive business plans, pitch development, intellectual property strategy, designated entity access, corporate structuring, business networks, and funding partners.",
       "The letter for Proof of Settlement Funds should be printed on the bank’s letterhead with contact information included. The original signature should be provided at the bottom. For questions, contact Upasana Sharma at usharma@iqcanada.ca.",
       "The funds required in Canadian dollars for family members vary based on the number of family members, ranging from $13,310 for one family member to $35,224 for seven family members, with an additional $3,586 for each extra family member.",
       "The minimum necessary income required according to the size of the family unit varies, ranging from $26,620 for one person to $70,448 for seven persons, with an additional $7,172 for each additional person.",
       "First, you must pitch your business idea to a designated organization and convince it to support your start-up company. Once a designated organization decides to support your business, it will send us a commitment certificate and give you a letter of support. The process to pitch your idea varies for each designated organization. You must contact the designated organization to find out what you must do to seek its support. If you meet all eligibility criteria, you can submit a completed application, which must include your letter of support.",
       "To get support from a venture capital fund or an angel investor group, you need a sizable investment from them. To get support from a business incubator, you must be accepted into a designated Canadian business incubator program. Getting into these programs is a competitive process with limited spots available. Contact a designated organization to find out how to get their support.",
       "First, the designated organization(s) that you seek support from will review the viability of your business proposal. Next, Immigration, Refugees and Citizenship Canada will review your immigration application, if you have received the necessary support to apply, to make sure you’re eligible to become a permanent resident. Finally, your application may also be subject to an independent peer review process. This is designed to protect against fraud and to ensure that your activities and those of the designated organization are in line with industry standards for this type of support.",
       "An essential person is critical to the proposed start-up. Without this person, the designated organization would not invest or support the proposal. If we refuse the application of an essential person, all related applicants will be refused.",
       "Angel investors usually invest their own money into a start-up business. Venture capital funds are limited partnerships that have a fund of pooled investment capital to invest in several companies. Also, angel investor groups invest in companies at an earlier stage of development, compared to venture capital funds. Minimum investment amounts are determined based on consultation with industry associations and represent typical investment amounts in each industry.",
       "No, you’re not required to invest any of your own money. The minimum investment required is an investment that comes from a designated Canadian venture capital fund or angel investor group.",
       "Receiving support from multiple designated venture capital funds or angel investor groups is known as syndication. If you have syndicated support, then all investment organizations involved in the syndication must be identified. Only one commitment certificate will be sent electronically to us and one letter of support will be provided to you. If a designated venture capital fund invests in your business, the minimum total investment amount is $200,000, even if a designated angel group also invests in your business. If at least one designated angel group but no designated venture capital fund invests in your business, the minimum total investment amount is $75,000.",
       "Those who meet the eligibility requirements must secure a minimum of $75,000 from a Canadian angel investor, or a minimum of $200,000 from a Canadian venture capital fund recognized by the government of Canada.",
       "To apply for a startup visa, fill out the application form, complete the document checklist, pay your application fees, and submit your application. Ensure that it is complete, signed, and includes all supporting documents. The envelope should be labeled 'start-up visa' and mailed to the appropriate address.",
       "IRCC states that the SUV program has a general success rate of above 75%.",
       "Here is a list of designated organizations, venture capital funds, angel investor groups, and business incubators: Venture Capital Funds: 7 Gate Ventures, Alt Ventures, Apex Innovative Investments Ventures, Arete Pacific Tech Ventures (VCC) Corp, BCF Ventures, BDC Venture Capital, Celtic House Venture Partners, Extreme Venture Partners LLP, First Fund, Golden Venture Partners Fund, LP, INP Capital, iNovia Capital Inc., Intrinsic Venture Capital, Lumira Ventures, Invest Nova Scotia, PRIVEQ Capital Funds, Real Ventures, Red Leaf Capital Corp, Relay Ventures, ScaleUp Venture Partners, Inc., Tangentia Ventures, Top Renergy Inc., Vanedge Capital Limited Partnership, Version One Ventures, WhiteHaven Venture, Westcap Management Ltd., Yaletown Venture Partners Inc., York Entrepreneurship Development Institute (YEDI) VC Fund,  Angel Investor Groups: Canadian International Angel Investors, Ekagrata Inc., Golden Triangle Angel Network, Keiretsu Forum Canada, Oak Mason Investments Inc., TenX Angel Investors Inc., VANTEC Angel Network Inc., York Angel Investors Inc., Business Incubators: Alacrity Foundation, Alberta Agriculture and Forestry, Agrivalue Processing Business Incubator, Food Processing Development Centre, Alberta IoT Association, Altitude Accelerator, Bhive, Biomedical Commercialization Canada Inc. (operating as Manitoba Technology Accelerator), Brilliant Catalyst, Creative Destruction Lab, DMZ Ventures, Empowered Startups Ltd., Extreme Innovations, Foresight Cleantech Accelerator Centre, Genesis Centre, Highline BETA Inc., Invest Nova Scotia, Innovate Calgary, Innovation Cluster - Peterborough and the Kawarthas, Innovation Factory, Interactive Niagara Media Cluster o/a Innovate Niagara, Intrinsic Innovations, Invest Ottawa, ISM Arts & Culture Ltd, Knowledge Park o/a Planet Hatch, LatAm Startups, Launch Academy - Vancouver, LaunchPad PEI Inc., Millworks Centre for Entrepreneurship, NEXT Canada, Niagara Business & Innovation Fund, North Forge East Ltd., North Forge Technology Exchange, Pacific Technology Ventures, Platform Calgary, Pycap, Real Investment Fund III L.P. o/a FounderFuel, Red Leaf Capital Corp, Roseview Global Incubator, Spark Commercialization and Innovation Centre, Spring Activator, The DMZ at Ryerson University, Think8 Global Institute, TiE Toronto, Toronto Business Development Centre (TBDC), Treefrog, TSRV Canada Inc. (operating as Techstars Canada), University of Toronto Entrepreneurship Hatchery, ventureLAB Innovation Centre, VIATEC, Waterloo Accelerator Centre, York Entrepreneurship Development Institute, YSpace (York University)",
       "To enhance the chances of a successful application, co-founders should ideally possess a strong entrepreneurial background, relevant industry experience, and a proven track record of success. Demonstrating a history of successfully launching and managing startups can significantly strengthen the application.",
       "Yes, a startup with a physical presence outside of Canada can still be eligible for the Startup Visa Program. However, the business must intend to operate in Canada, and a significant portion of its operations and job creation should occur within the country.",
       "If there are changes in the ownership structure or leadership team after receiving the startup visa, it is crucial to inform the designated organization that provided the Letter of Support. Depending on the nature of the changes, the government may reassess the eligibility of the startup.",
       "Startups that have received funding or support from non-designated organizations are still eligible to apply for the Startup Visa Program. However, the primary support for the visa application should come from a designated organization.",
       "The business plan should outline the startup's goals, market strategy, financial projections, and the innovative aspects of the business. A comprehensive and well-researched business plan strengthens the application.",
       "The Startup Visa Program does not have specific provisions for intellectual property rights. Entrepreneurs are responsible for protecting their intellectual property, and standard Canadian laws and processes apply.",
       "While the focus is on the initially approved business idea, entrepreneurs can make changes with the approval of the designated organization. Major changes may require reassessment by immigration authorities.",
       "Provinces and territories do play a role, and choosing a specific location can have advantages. Some provinces may have additional programs or incentives for startups. Researching regional opportunities is recommended.",
       "Yes, entrepreneurs who have previously applied for other Canadian immigration programs can still apply for the Startup Visa Program, provided they meet the program's requirements.",
       "The Startup Visa Program aims to contribute to diversity and inclusion by attracting entrepreneurs from around the world. The program values a diverse range of skills, experiences, and perspectives in the Canadian entrepreneurial ecosystem.",
       "Startup visa holders can pursue additional education or training while running their business in Canada. However, the primary focus should be on actively managing and growing the startup.",
       "Designated organizations may continue to provide support in the post-approval phase, offering resources, networking opportunities, and guidance as entrepreneurs establish and expand their businesses.",
       "While there are no specific tax incentives tied to the Startup Visa Program, startups in Canada may benefit from general tax incentives and credits available to businesses.",
       "In such cases, the startup may need to find an alternative designated organization to maintain eligibility. Communication with immigration authorities is essential to address any changes in the support structure.",
       "Startup visa holders can sell their business and maintain their Permanent Residency status in Canada. However, they should continue to meet the program requirements, and any significant changes should be communicated to immigration authorities.",
       "The startup should demonstrate its potential for scalability and growth through a well-defined business model, a large addressable market, and innovative solutions. Key benchmarks may include market traction, revenue projections, and a clear plan for expansion.",
       "Canada's startup ecosystem plays a crucial role in supporting and integrating startup visa holders. It provides access to mentorship, networking opportunities, funding, and collaborative environments, fostering a supportive community for entrepreneurs.",
       "The Startup Visa Program accommodates businesses in emerging or unconventional industries, recognizing that innovation can arise in various sectors. The key is to demonstrate the potential for economic growth, job creation, and innovation in the chosen industry.",
       "There are generally no specific restrictions on the use of funds obtained through designated organizations. However, funds should be used to support the growth and development of the startup, aligning with the objectives outlined in the business plan.",
       "Yes, a startup visa holder can bring in foreign partners or collaborators to contribute to the business. Collaborative efforts can enhance the startup's capabilities and increase its chances of success.",
       "If challenges or setbacks arise in meeting job creation targets, communication with the designated organization and immigration authorities is essential. Providing a revised plan or seeking guidance on overcoming obstacles may be necessary.",
       "The program acknowledges that business pivots may be necessary for a startup's survival or success. Entrepreneurs should communicate such changes to the designated organization and immigration authorities, ensuring transparency in the process.",
       "Yes, a startup visa holder can apply for additional funding or support from designated organizations after the initial approval. The key is to demonstrate the need for additional resources to achieve the business's objectives.",
       "Startup visa holders are subject to the standard tax obligations for businesses operating in Canada. They should comply with federal and provincial tax regulations, and seeking professional advice is recommended.",
       "The program evaluates startups based on their potential to contribute to the Canadian economy. Entrepreneurs should highlight unique value propositions, market differentiators, and strategies to navigate competition and saturation in their industries.",
       "While the primary focus should be on the designated business, entrepreneurs can participate in other entrepreneurial activities or ventures outside of their designated business. However, they should ensure that these activities do not compromise their commitment to the startup visa requirements.",
       "If a designated organization disagrees with the direction or management of the startup after providing a Letter of Support, communication and collaboration are crucial. Open dialogue and, if needed, involving immigration authorities can help resolve disputes.",
       "Startups in the healthcare or biotechnology sectors should meet the general program requirements. Additionally, they should demonstrate how their innovations contribute to advancements in these fields and align with the program's objectives.",
       "Yes, a startup visa holder can hire foreign workers to contribute to the growth of their business in Canada. However, the hiring process should comply with Canadian immigration and labor regulations.",
       "If a startup is acquired by another company, the new ownership structure should be communicated to the designated organization and immigration authorities. The startup visa holder may need to provide updated information and possibly undergo reassessment.",
       "There are no specific restrictions on the types of investments that designated organizations can make in startups. However, the investments should align with the objectives of the Startup Visa Program.",
       "The program addresses concerns related to the sustainability and long-term viability of a startup through the ongoing support provided by designated organizations. Regular communication and updates on the business's progress are essential.",
       "Yes, a startup visa holder can apply for additional permits for international employees who are crucial to the business's success. This may involve obtaining work permits for key personnel.",
       "Various resources are available to help startup visa holders integrate into the Canadian business community and culture. These may include networking events, mentorship programs, and support services offered by designated organizations and the broader startup ecosystem.",
       "The Startup Visa Program encourages partnerships with research institutions or universities for innovative startups. Collaborations can enhance the innovative nature of the business and contribute to the growth of the Canadian knowledge economy.",
       "Yes, a startup visa holder can relocate their business to a different province or territory within Canada after obtaining Permanent Residency. However, they should inform immigration authorities and ensure compliance with any provincial regulations or incentives.",
       "The startup community plays a vital role in supporting newcomers through networking events and collaborative initiatives. These events facilitate connections, knowledge-sharing, and the integration of startup visa holders into the broader entrepreneurial ecosystem.",
       "The program employs rigorous screening processes and verification measures to protect against fraud or misrepresentation in the startup visa application process. Applicants are expected to provide accurate and truthful information, and any fraudulent activity may result in application rejection or revocation of status.",
       "The program recognizes the importance of supporting startups in rural or remote areas and encourages their participation. Designated organizations may play a key role in providing resources, mentorship, and connections to help startups thrive outside major urban centers.",
       "Startup visa holders may take temporary leaves from their business without jeopardizing their immigration status, but it's advisable to inform designated organizations and immigration authorities to ensure compliance with program requirements.",
       "Dispute resolution between co-founders or with designated organizations during the application process may involve communication, mediation, or legal intervention if necessary. Clear documentation and agreements can help prevent and address disputes.",
       "The program acknowledges external factors like economic downturns or market fluctuations that may impact startups. Flexibility and communication with designated organizations and immigration authorities are essential during challenging times.",
       "Startup visa holders can sponsor skilled international workers through other immigration programs, such as the Temporary Foreign Worker Program or the International Mobility Program. Compliance with relevant immigration regulations is necessary.",
       "The program encourages collaboration and knowledge-sharing among startups within the same industry or sector through networking events, industry associations, and initiatives that foster a culture of innovation and cooperation.",
       "Criteria for demonstrating the potential for international market expansion include a well-defined market entry strategy, global market demand, and a clear plan for scaling operations beyond Canada. Designated organizations may provide guidance on meeting these criteria.",
       "Startup visa holders can access government grants or funding programs available to businesses in Canada. These programs may provide additional financial support to further facilitate business growth.",
       "While there are no specific provisions in the program, standard legal and business practices apply for the protection of sensitive business information and trade secrets. Entrepreneurs should implement appropriate measures to safeguard their intellectual property.",
       "Yes, startup visa holders can participate in accelerator programs or business competitions to further develop their business. Such participation can enhance visibility, attract investment, and provide valuable resources for growth.",
       "The Startup Visa Program actively fosters innovation in traditional industries or sectors by welcoming startups with innovative approaches, technologies, or business models that contribute to the modernization and competitiveness of these industries.",
       "Startup visa holders can apply for a work permit for a key employee who is not eligible for permanent residency. The hiring process should comply with Canadian immigration regulations.",
       "The program assesses the impact of a startup on the local community and economy by considering factors such as job creation, economic contributions, and community engagement. Designated organizations may play a role in evaluating these impacts.",
       "Initiatives and resources exist to support startups led by underrepresented groups, including women or minorities. These may include mentorship programs, funding opportunities, and networking events specifically tailored to promote diversity in entrepreneurship.",
       "Yes, a startup visa holder can apply for additional funding from other sources after the initial investment from a designated organization. Diversifying funding sources can contribute to the financial stability and growth of the startup.",
       "If a startup pivots its business model or focus after receiving approval, it should communicate these changes to designated organizations and immigration authorities. Transparency and cooperation are key to ensuring continued compliance with program requirements.",
       "The documents required are: 1. Business plan, 2. Letter of Support, 3. Commitment certificate, 4. Proof of Investment, 5. Language proficiency, 6. Education Credential Assessment(ECA), 7. Net worth statements, 8. Police clearance certificates, 9. Passport and travel documents, 10. Passport Sized Photographs, 11. Application forms inclcuding the IMM 5760 Business Immigration Start-Up Business Class., 12. Proof of Settlement funds, 13. Medical examination, 14. Travel History, 15. Resume/CV, 16. Family information, 17. Additional documents. It's important to note that the specific requirements may vary, and it's advisable to check the official website of Immigration, Refugees, and Citizenship Canada (IRCC) for the most up-to-date information and detailed instructions for the Start-Up Visa program."
    ]

    question_db = [preprocess_text(question) for question in startup_questions]
    answer_db = startup_answers
    return question_db, answer_db

question_db, answer_db = create_startup_visa_chatbot()
chatbot_model = create_chatbot(question_db, answer_db)

@app.route('/')
def index():
    return render_template("zz.html")

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    
    if user_input.lower() in ['bye', 'exit', 'quit']:
        response = "Goodbye!"
    elif user_input.lower() in ['hi', 'hello', 'hey']:
        response = "Hello! How can I assist you today?"
    else:
        response = chatbot_model(user_input)

    return jsonify({'response': response})

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        # Get text from the request
        text = request.form.get('text')

        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Set properties (optional)
        engine.setProperty('rate', 150)  # Speed of speech

        # Speak the text
        engine.say(text)
        engine.runAndWait()

        return jsonify({'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    app.run(debug=False)
