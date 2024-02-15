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
import pyodbc
from datetime import datetime
db_connection_string = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=103.145.51.250;"
    "Database=PMO360_DB;"
    "UID=PMOlogbook_Usr;"
    "PWD=PMO_log360!x4;"
)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    return ' '.join(tokens)

def gpt(prompt,ch):
  response= client.chat.completions.create(
         model="gpt-3.5-turbo-0125",
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
        
    {"role": "user", "content": "What are the criteria for a qualifying business in the Start-up Visa Program?"},

{"role": "assistant", "content": "Criteria for a qualifying business in the Start-up Visa Program:\n\n\
    1. Formation: Applicants required to start a qualifying business\n\
    2. Ownership structure compliance with Minister's percentages\n\
    3. Active and ongoing management from within Canada\n\
    4. Essential operations conducted in Canada, including:\n\
        - Physical location, facilities, equipment, inventory, warehousing\n\
        - Management information systems, staff\n\
    5. Incorporation in Canada\n\
    6. Intent to meet criteria upon permanent residence\n\
    7. Qualifying business ownership structure:\n\
        - Each applicant holds 10% or more voting rights\n\
        - Applicants and designated entity jointly hold over 50% voting rights\n\
    8. Commitment from designated entities:\n\
        - Angel investor group: $75,000 investment\n\
        - Venture capital fund: $200,000 investment\n\
        - Business incubator acceptance\n\
    9. Intent to incorporate if not yet incorporated. Any more questions on this?"},
    
    {"role": "user", "content": "Can you elaborate on the ownership structure criteria for a qualifying business under the Start-up Visa Program?"},

{"role": "assistant", "content": "Certainly! The ownership structure criteria for a qualifying business involve:\n\n\
    1. Each applicant holding 10% or more of voting rights\n\
    2. Jointly holding more than 50% of total voting rights with the designated entity\n\
    3. Ensures significant control and participation of qualified participants. Need more details or clarification?"},

{"role": "user", "content": "What happens if the applicant's business is not yet incorporated at the time of application?"},

{"role": "assistant", "content": "If the applicant's business is not yet incorporated at the time of application for the Start-up Visa Program:\n\n\
    1. Business may still be considered qualifying in some cases\n\
    2. Intent to incorporate in Canada required after receiving permanent resident visa\n\
    3. Officer satisfaction necessary if the initial criteria are not met upon application. Anything else you'd like to know?"},

{"role": "user", "content": "How does the Start-up Visa Program define an 'investment' in the applicant's business?"},

{"role": "assistant", "content": "In the Start-up Visa Program, an 'investment' in the applicant's business is defined as:\n\n\
    1. Buying shares or other forms of equity\n\
    2. Convertible debt or debenture not permitted\n\
    3. Emphasizes tangible ownership stake. Need further clarification or details?"},

{"role": "user", "content": "What role do designated entities play in confirming commitments for a qualifying business?"},

{"role": "assistant", "content": "Designated entities play a vital role in confirming commitments for a qualifying business in the Start-up Visa Program:\n\n\
    1. Angel investor groups: Confirming at least $75,000 investment\n\
    2. Venture capital funds: Confirming at least $200,000 investment\n\
    3. Business incubators: Confirming acceptance into their program\n\
    4. Officers ensure that only qualified participants own over 50% voting rights. Any more questions on this process?"},
    
{"role": "user", "content": "Can you explain the significance of the commitment certificate and letter of support in the Start-up Visa application process?"},

{"role": "assistant", "content": "Certainly! The commitment certificate and letter of support are crucial in the Start-up Visa application process:\n\n\
    1. Demonstrate support from designated entities\n\
    2. Issued by designated entities, less than 6 months old\n\
    3. Sent directly to Immigration, Refugees and Citizenship Canada (IRCC)\n\
    4. Commitment certificate conditions:\n\
        - Issued by designated entities\n\
        - No fees for reviewing or assessing business proposal\n\
        - Includes term sheet/client agreement, investment details, and services\n\
        - Information on each applicant (up to 5)\n\
        - Describes relationship between applicant and investment team\n\
        - Details applicant's role and tasks in the business\n\
    5. Letter of support consistency check with commitment certificate\n\
    6. No limit to supporting entities, but only 1 commitment certificate for syndication\n\
    7. Venture capital fund leads in syndication with $200,000 minimum investment. Need more details or specific questions?"},

{"role": "user", "content": "What role does the lead designated entity play in the commitment certificate process?"},

{"role": "assistant", "content": "The lead designated entity plays a crucial role in the commitment certificate process in the Start-up Visa Program:\n\n\
    1. Responsible for submitting the commitment certificate to IRCC\n\
    2. Manages the coordination and submission process for syndication\n\
    3. Ensures consistency and completeness in the details provided by participating designated entities\n\
    4. Critical for streamlined communication with immigration authorities. Any more questions on this role?"},

{"role": "user", "content": "How does the Start-up Visa Program handle syndication when multiple designated entities are involved?"},

{"role": "assistant", "content": "In the Start-up Visa Program, syndication with multiple designated entities involves:\n\n\
    1. Only 1 commitment certificate issued for syndication\n\
    2. Contains details and requirements of all participating designated entities\n\
    3. No limit to the number of supporting entities\n\
    4. Venture capital fund leads in syndication with $200,000 minimum investment\n\
    5. If no venture capital funds, the lead must be the angel investor group with a $75,000 minimum investment. Need further clarification or details?"},

{"role": "user", "content": "How are spouses or common-law partners involved in the Start-up Visa Program?"},

{"role": "assistant", "content": "Spouses or common-law partners in the Start-up Visa Program:\n\n\
    1. If part of the entrepreneurial team, must be on the commitment certificate\n\
    2. Processed as a business applicant under Start-up Visa (SUV), including both essential and non-essential roles\n\
    3. If owning shares but not part of the team, shares can't be counted for the ownership percentage (requirement > 50%)\n\
    4. If not on the team, may immigrate as a family member if meeting all requirements. Need more information or specific questions?"},

{"role": "user", "content": "What happens if a spouse or common-law partner is part of the entrepreneurial team but not on the commitment certificate?"},

{"role": "assistant", "content": "If a spouse or common-law partner is part of the entrepreneurial team but not on the commitment certificate in the Start-up Visa Program:\n\n\
    1. Not fulfilling program requirements\n\
    2. Essential for spouses or partners to be listed on the commitment certificate for processing as business applicants\n\
    3. Their involvement is crucial for assessing the overall team composition. Any further questions or clarification needed?"},

{"role": "user", "content": "Can you explain the role of spouses or common-law partners who own shares but are not part of the entrepreneurial team?"},

{"role": "assistant", "content": "Certainly! Spouses or common-law partners who own shares but are not part of the entrepreneurial team in the Start-up Visa Program:\n\n\
    1. Their shares cannot be counted for the ownership percentage requirement (> 50%)\n\
    2. Highlights the distinction between ownership and active participation\n\
    3. Important for assessing the true involvement of team members in the business. Any more questions on this or related topics?"},

{"role": "user", "content": "What information is essential for designated business incubators in the Start-up Visa Program?"},

{"role": "assistant", "content": "Essential information for designated business incubators in the Start-up Visa Program includes:\n\n\
    1. Identifying the applicant\n\
    2. Confirming acceptance into the business incubator program\n\
    3. Specifying the period of time for the applicant's involvement if established\n\
    4. Describing the nature of business operations by the applicant\n\
    5. Defining the applicant's role in the business\n\
    6. Specifying control over intellectual property and assets brought to the business\n\
    7. Describing the legal and financial structure of the business\n\
    8. Identifying roles of other persons with expected interests\n\
    9. Confirming due diligence assessment of the applicant and the business\n\
    10. Specifying terms and conditions applicable to the business incubator program\n\
    11. Confirming the business as a qualifying business\n\
    12. Details of services provided and associated fees. Need more details or have other questions?"},

{"role": "user", "content": "How are due diligence assessments handled by designated angel investor groups and venture capital funds in the Start-up Visa Program?"},

{"role": "assistant", "content": "Due diligence assessments by designated angel investor groups and venture capital funds in the Start-up Visa Program involve:\n\n\
    1. Confirming the applicant's identity and nature of business operations\n\
    2. Assessing the applicant's role in the business\n\
    3. Verifying control over intellectual property and assets\n\
    4. Reviewing the legal and financial structure of the business\n\
    5. Identifying roles of other persons with expected interests\n\
    6. Confirming a thorough assessment, including provenance of funds\n\
    7. Ensuring investment in a qualifying business\n\
    8. Specifying investment amount and applicable terms and conditions\n\
    9. Detailing services and associated fees. Any more questions on this process?"},

{"role": "user", "content": "How are settlement funds assessed in the Start-up Visa Program?"},

{"role": "assistant", "content": "Settlement funds in the Start-up Visa Program are assessed based on the following criteria:\n\n\
    1. Must be available and transferable\n\
    2. Unencumbered by debts or other obligations\n\
    3. Sufficient to support initial establishment in Canada\n\
    4. Excludes any investment made by a designated entity into the business\n\
    5. Official letters required from banks or financial institutions\n\
    6. Letters must be on financial institution’s letterhead\n\
    7. Include contact information, applicant’s name, and list outstanding debts\n\
    8. Provide details for each bank and investment account\n\
        - Account numbers\n\
        - Date opened\n\
        - Current balance\n\
        - Average balance for the past 6 months\n\
    9. Funds determined based on family size and 50% of the current low-income cut-off for urban areas with a population of 500,000 or more. Need more details or specific questions?"},

{"role": "user", "content": "What is the importance of language proficiency in the Start-up Visa application?"},

{"role": "assistant", "content": "Language proficiency is crucial in the Start-up Visa application for the following reasons:\n\n\
    1. Required proof of language skills approved by IRCC\n\
    2. Minimum level of 5 in the Canadian Language Benchmark in English or the Niveaux de compétence linguistique canadiens in French for each of the 4 language skill areas\n\
    3. Test results must be less than 2 years old on the date of application\n\
    4. Ensures effective communication and integration into the Canadian business environment. Any more questions or specific concerns?"},

{"role": "user", "content": "Can you provide more details on the specific criteria for settlement funds in the Start-up Visa Program?"},

{"role": "assistant", "content": "Certainly! Specific criteria for settlement funds in the Start-up Visa Program include:\n\n\
    1. Availability and transferability\n\
    2. Being unencumbered by debts or other obligations\n\
    3. Meeting the required amount to support initial establishment in Canada\n\
    4. Excluding any investment made by a designated entity into the business\n\
    5. Can you provide more details or ask about a specific aspect?"},

{"role": "user", "content": "How are settlement funds determined based on family size in the Start-up Visa Program?"},

{"role": "assistant", "content": "Settlement funds in the Start-up Visa Program are determined based on family size:\n\n\
    1. Utilizing 50% of the current low-income cut-off for urban areas with a population of 500,000 or more\n\
    2. Ensures adequate financial support for both accompanying and non-accompanying family members\n\
    3. The calculation aims at providing a fair and reasonable financial threshold\n\
    4. Need more details or specifics on family size considerations?"},

{"role": "user", "content": "What documents are required to prove settlement funds in the Start-up Visa application?"},

{"role": "assistant", "content": "Documents required to prove settlement funds in the Start-up Visa application include:\n\n\
    1. Official letters from banks or financial institutions\n\
    2. Must be printed on the financial institution’s letterhead\n\
    3. Include contact information, applicant’s name, and list outstanding debts\n\
    4. Details for each current bank and investment account:\n\
        - Account numbers\n\
        - Date each account was opened\n\
        - Current balance of each account\n\
        - Average balance for the past 6 months\n\
    5. Ensures a comprehensive and verifiable assessment of available funds. Anything else you'd like to know?"},

{"role": "user", "content": "How often are language proficiency test results required in the Start-up Visa application process?"},

{"role": "assistant", "content": "Language proficiency test results in the Start-up Visa application process:\n\n\
    1. Must be less than 2 years old on the date of application\n\
    2. Ensures recent and up-to-date assessment of language skills\n\
    3. Allows immigration authorities to evaluate the applicant's current language proficiency\n\
    4. Maintains the relevance of language testing. Need more information or have specific questions on language proficiency?"},



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
        store_user_information(response,user_input)
        return response

    return chatbot

def store_user_information(response,user):
        current_datetime = datetime.utcnow()
        formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        user_info_conn, user_info_cursor = get_user_info_database_connection()
        user_info_cursor.execute('''
            INSERT INTO Iq_chathist (User_input,Bot_response,Date) VALUES (?, ?, ?)
        ''', (user,response,formatted_date))
        user_info_conn.commit()
        user_info_conn.close()

def get_user_info_database_connection():
    user_info_conn = pyodbc.connect(db_connection_string)
    user_info_cursor = user_info_conn.cursor()
    return user_info_conn, user_info_cursor

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
