import functionalities as fun
from langchain.memory import ConversationBufferMemory

if __name__ == "__main__":
    memory = ConversationBufferMemory()
    
    while True:
        user_query = input("Write your symptoms or query: ").strip().lower()
        
        # Handle greetings and goodbyes
        response = fun.handle_greetings_and_goodbyes(user_query)
        if response:
            print("\nBot:", response)
            if user_query in ["bye", "goodbye", "exit", "quit", "see you"]:
                break
            continue
        
        # Classify the user query type
        if not fun.classify_query(user_query): 
            print("\nBot: I'm a medical assistant and can only help with health-related questions.")
            continue
        
        query_type = fun.classify_query_type(user_query)
        
        if query_type == "unknown":
            print("\nBot: Can you specify more clearly, please?")
            continue
        
        elif query_type == "treatment":
            treatment = fun.generate_treatment(user_query, memory)
            print("\nBot: Here is the recommended treatment:\n", treatment)
            memory.save_context({"input": user_query}, {"output": treatment})
            continue
        
        elif query_type == "diagnosis":
            follow_up_questions = fun.generate_best_questions(user_query, memory)
            user_responses = []

            for i, question in enumerate(follow_up_questions, start=1):
                user_answer = input(f"Bot ({i}/5): {question}\nYour response: ")
                user_responses.append(f"Q{i}: {question} | A{i}: {user_answer}")
                memory.save_context({"input": question}, {"output": user_answer})

            combined_responses = "\n".join(user_responses)
            final_diagnosis = fun.generate_final_diagnosis(user_query, combined_responses, memory)

            print("\nBot: Here is your final diagnosis:\n", final_diagnosis)
            memory.save_context({"input": combined_responses}, {"output": final_diagnosis})
            continue
        
        elif query_type == "general_medical":
            medical_info = fun.generate_general_medical_info(user_query)
            print("\nBot:", medical_info)
            memory.save_context({"input": user_query}, {"output": medical_info})
            continue
        
        elif query_type == "medicine_info":
            medicine_info = fun.generate_medicine_info(user_query)
            print("\nBot:", medicine_info)
            memory.save_context({"input": user_query}, {"output": medicine_info})
            continue
        
        elif query_type == "health_advice":
            health_advice = fun.generate_health_advice(user_query)
            print("\nBot:", health_advice)
            memory.save_context({"input": user_query}, {"output": health_advice})
            continue
        
        elif query_type == "emergency_advice":
            emergency_tips = fun.generate_emergency_advice(user_query)
            print("\nBot:", emergency_tips)
            memory.save_context({"input": user_query}, {"output": emergency_tips})
            continue
