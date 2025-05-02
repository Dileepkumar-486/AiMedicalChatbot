from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

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

RESTRICTED_TOPICS = [
    "sports", "movies", "technology", "history", "politics", "science", "entertainment",
    "business", "cooking", "travel", "education", "finance", "gaming", "music", "art",
    "relationships", "celebrities", "stocks", "astronomy", "math", "physics", "weather",
    "shopping", "cars", "fashion", "news"
]

treatment_keywords = [
    "cure", "treatment", "medicine", "remedy", "medication", "therapy", "how to treat", 
    "how to cure", "how to heal", "what is the remedy for", "best treatment for", 
    "treatment options for", "cure for", "remedies for", "how to fix", "how to relieve", 
    "healing for", "medicine for", "what should I take for", "what should I do for", 
    "how to manage", "how to alleviate", "how to improve", "what is the solution for", 
    "how to prevent", "how to recover from", "how to stop", "how to reduce"
]

def handle_greetings_and_goodbyes(user_input):
    """Check if input is a greeting or goodbye and return an appropriate response."""
    user_input = user_input.lower().strip()
    
    GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    GOODBYES = ["bye", "goodbye", "exit", "quit", "see you"]

    if user_input in GREETINGS:
        return "Hello! How can I assist you with your medical concerns today?"
    elif user_input in GOODBYES:
        return "Take care! Stay healthy."
    
    return None

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

def is_medical_query(user_input):
    search_results = db.similarity_search(user_input, k=1)
    return bool(search_results)

def contains_restricted_topic(user_input):
    return any(topic in user_input.lower() for topic in RESTRICTED_TOPICS)

# def generate_best_questions(user_input):
#     prompt = PromptTemplate(template=follow_up_prompt_template, input_variables=["user_input"])
#     llm = load_llm(huggingface_repo_id)
#     question_chain = LLMChain(llm=llm, prompt=prompt)
#     return question_chain.invoke({"user_input": user_input})["text"].strip().split("\n")

def generate_best_questions(user_input, conversation_history):
    prompt = PromptTemplate(template=follow_up_prompt_template, input_variables=["user_input", "conversation_history"])
    llm = load_llm(huggingface_repo_id)
    question_chain = LLMChain(llm=llm, prompt=prompt)
    return question_chain.invoke({"user_input": user_input, "conversation_history": conversation_history})["text"].strip().split("\n")


def generate_final_diagnosis(user_input, user_responses):
    prompt = PromptTemplate(template=diagnosis_prompt_template, input_variables=["user_input", "user_responses"])
    llm = load_llm(huggingface_repo_id)
    diagnosis_chain = LLMChain(llm=llm, prompt=prompt)
    return diagnosis_chain.invoke({"user_input": user_input, "user_responses": user_responses})["text"].strip()

def generate_treatment(user_input):
    prompt = PromptTemplate(template=treatment_prompt_template, input_variables=["user_input"])
    llm = load_llm(huggingface_repo_id)
    treatment_chain = LLMChain(llm=llm, prompt=prompt)
    return treatment_chain.invoke({"user_input": user_input})["text"].strip()


def classify_query(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class == 1  # True if medical, False otherwise

if __name__ == "__main__":
    while True:
        user_query = input("Write your symptoms or query: ").strip().lower()

        response = handle_greetings_and_goodbyes(user_query)
        if response:
            print("\nBot:", response)
            if user_query in ["bye", "goodbye", "exit", "quit", "see you"]:
                break
            continue

        if not classify_query(user_query): 
            print("\nBot: I'm a medical assistant and can only help with health-related questions.")
            continue

        if any(keyword in user_query.lower() for keyword in ["cure", "treatment", "medicine", "remedy"]):
            treatment = generate_treatment(user_query)
            print("\nBot: Here is the recommended treatment:\n", treatment)
            continue

        follow_up_questions = generate_best_questions(user_query)
        user_responses = []

        for i, question in enumerate(follow_up_questions, start=1):
            user_answer = input(f"Bot ({i}/5): {question}\nYour response: ")
            user_responses.append(f"Q{i}: {question} | A{i}: {user_answer}")

        combined_responses = "\n".join(user_responses)
        final_diagnosis = generate_final_diagnosis(user_query, combined_responses)

        print("\nBot: Here is your final diagnosis:\n", final_diagnosis)
