from langchain_huggingface import HuggingFaceEndpoint # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from huggingface_hub import login
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
from transformers import BertTokenizer, BertForSequenceClassification
import torch


HF_TOKEN = "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"
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = r"D:\projects\chat\database"

# Load the trained BERT medical classifier (PyTorch version)
MODEL_PATH = r"D:\projects\chat\bert_medical_classifier_pytorch"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

login(token=HF_TOKEN)

def get_embedding_model():
  embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  return embedding_model

embedding_model = get_embedding_model()

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        streaming=True,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    return llm

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)



follow_up_prompt_template = """
You are a medical chatbot helping users diagnose symptoms.
Given the user's symptom, generate the **five most important** follow-up questions, one per line.

User Symptom: {user_input}

Best 5 follow-up questions:
"""

diagnosis_prompt_template = """
You are a medical assistant. Based on the patient's symptoms and their responses to five key follow-up questions,
provide a concise, accurate diagnosis or recommendation.

User Symptom: {user_input}
User Responses:
{user_responses}

Final Diagnosis or Recommendation:
"""

treatment_prompt_template = """
You are a medical expert providing treatment recommendations.
Based on the user's symptom, provide the best possible treatment options, home remedies, or medications.

User Symptom: {user_input}


Best Treatment or Cure:
"""


GREETINGS = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
    "howdy", "what's up", "sup", "yo", "greetings"
]

GOODBYES = [
    "bye", "goodbye", "exit", "quit", "see you", "take care", "farewell", 
    "later", "good night", "talk soon"
]

treatment_keywords = ["cure", "treatment", "medicine", "remedy", "medication", "therapy", "healing", "fix", "relieve", "alleviate"]

diagnosis_keywords = ["i am suffering", "i have", "my symptoms are", "what is wrong with me", "diagnose", "diagnosis", "symptoms"]

general_medical_keywords = ["what is", "causes of", "symptoms of", "explain", "information on", "understanding"]

medicine_info_keywords = ["side effects", "uses of", "dosage of", "ingredients", "how much", "take for", "medication"]

health_advice_keywords = ["is it safe to", "can i", "should i", "recommend", "advice on", "tips for"]

emergency_triggers = ["cut", "bleeding", "fainted", "suicide", "heart attack", "stroke", "choking", "allergic reaction","emergency", "first aid","first aid", "what to do if", "how to handle", "what happens if", "urgent" ,"care", "urgent care"]

# Function to get top 3 most relevant documents from FAISS DB
def search_faiss_db(query):
    results = db.similarity_search(query, k=1)  
    return results

def handle_greetings_and_goodbyes(user_input):
    """Check if input is a greeting or goodbye and return an appropriate response."""
    user_input = user_input.lower().strip()
    words = user_input.split()  # Split input into words

    if user_input in GREETINGS:
        return "Hello! How can I assist you with your medical concerns today?"
    
    # Check if any word in input matches a goodbye
    if any(word in GOODBYES for word in words):
        return "Take care! Stay healthy."
    
    return None

def generate_best_questions(user_input):
    prompt = PromptTemplate(template=follow_up_prompt_template, input_variables=["user_input"])
    llm = load_llm(huggingface_repo_id)
    question_chain = LLMChain(llm=llm, prompt=prompt)
    return question_chain.invoke({"user_input": user_input})["text"].strip().split("\n")

def generate_final_diagnosis(user_input, user_responses):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])
    prompt = PromptTemplate(template=diagnosis_prompt_template, input_variables=["user_input", "user_responses", "faiss_text"])
    llm = load_llm(huggingface_repo_id)
    diagnosis_chain = LLMChain(llm=llm, prompt=prompt)
    return diagnosis_chain.invoke({"user_input": user_input, "user_responses": user_responses, "faiss_text": faiss_text})["text"].strip()

def generate_treatment(user_input):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])
    prompt = PromptTemplate(template=treatment_prompt_template, input_variables=["user_input", "faiss_text"])
    llm = load_llm(huggingface_repo_id)
    treatment_chain = LLMChain(llm=llm, prompt=prompt)
    return treatment_chain.invoke({"user_input": user_input, "faiss_text": faiss_text})["text"].strip()

def classify_query(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class == 1  # True if medical, False otherwise

# def classify_query(query):
#     inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     predicted_class = torch.argmax(outputs.logits, dim=1).item()
#     return predicted_class == 1  # True if medical, False otherwise    

def generate_general_medical_info(user_input):
    # Search the FAISS database for top 3 most relevant documents
    faiss_results = search_faiss_db(user_input)
    
    # Access the page_content of each FAISS result (Document objects)
    faiss_text = "\n".join([result.page_content for result in faiss_results])
    
    # Generate general medical info using FAISS results and user input
    prompt = PromptTemplate(
        template=(
            "Explain the following medical condition based on the relevant documents: '{faiss_text}'\n"
            "User input: {user_input}\n"
            "Provide a clear and concise explanation of the medical condition."
        ),
        input_variables=["user_input", "faiss_text"]
    )
    
    # Load the LLM
    llm = load_llm(huggingface_repo_id)
    info_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Invoke the LLM chain to generate the general medical info
    result = info_chain.invoke({
        "user_input": user_input, 
        "faiss_text": faiss_text
    })
    
    # Return the LLM-generated explanation
    return result["text"].strip()

def generate_medicine_info(user_input):
    # Search the FAISS database for top 3 most relevant documents
    faiss_results = search_faiss_db(user_input)
    
    # Access the page_content of each FAISS result (Document objects)
    faiss_text = "\n".join([result.page_content for result in faiss_results])
    
    # Generate medicine info using FAISS results and user input
    prompt = PromptTemplate(
        template=(
            "Provide detailed information about the following medicine based on the documents: '{faiss_text}'\n"
            "User input: {user_input}\n"
            "Provide clear and structured information about the medicine."
        ), 
        input_variables=["user_input", "faiss_text"]
    )
    
    # Load the LLM
    llm = load_llm(huggingface_repo_id)
    med_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Invoke the LLM chain to generate the medicine info
    result = med_chain.invoke({
        "user_input": user_input, 
        "faiss_text": faiss_text
    })
    
    # Return the LLM-generated medicine information
    return result["text"].strip()

def generate_health_advice(user_input):
    # Search the FAISS database for top 3 most relevant documents
    faiss_results = search_faiss_db(user_input)
    
    # Access the page_content of each FAISS result (Document objects)
    faiss_text = "\n".join([result.page_content for result in faiss_results])
    
    # Generate health advice using FAISS results and user input
    prompt = PromptTemplate(
        template=(
            "Give health advice based on the following user query: '{user_input}'\n"
            "And the relevant information from these documents: '{faiss_text}'\n"
            "Provide clear, actionable health advice."
        ),
        input_variables=["user_input", "faiss_text"]
    )
    
    # Load the LLM
    llm = load_llm(huggingface_repo_id)
    advice_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Invoke the LLM chain to generate the health advice
    result = advice_chain.invoke({
        "user_input": user_input, 
        "faiss_text": faiss_text
    })
    
    # Return the LLM-generated health advice
    return result["text"].strip()

def generate_emergency_advice(user_input):
    # Search the FAISS database for top 3 most relevant documents
    faiss_results = search_faiss_db(user_input)
    
    # Access the page_content of each FAISS result (Document objects)
    faiss_text = "\n".join([result.page_content for result in faiss_results])
    
    # Generate emergency advice using FAISS results and user input
    prompt = PromptTemplate(
        template=(
            "Provide emergency advice based on the user query: '{user_input}'\n"
            "And the relevant emergency information from documents: '{faiss_text}'\n"
            "Provide clear and actionable emergency steps or advice."
        ), 
        input_variables=["user_input", "faiss_text"]
    )
    
    # Load the LLM
    llm = load_llm(huggingface_repo_id)
    emergency_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Invoke the LLM chain to generate the emergency advice
    result = emergency_chain.invoke({
        "user_input": user_input, 
        "faiss_text": faiss_text
    })
    
    # Return the LLM-generated emergency advice
    return result["text"].strip()



def classify_query_type(query):
    query = query.lower()

    if any(keyword in query for keyword in emergency_triggers):
        return "emergency_advice"
    elif any(keyword in query for keyword in treatment_keywords):
        return "treatment"
    elif any(keyword in query for keyword in diagnosis_keywords):
        return "diagnosis"
    elif any(keyword in query for keyword in general_medical_keywords):
        return "general_medical"
    elif any(keyword in query for keyword in medicine_info_keywords):
        return "medicine_info"
    elif any(keyword in query for keyword in health_advice_keywords):
        return "health_advice"
    else:
        return "unknown"

