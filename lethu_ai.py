import ollama

def chat():
    print("Welcome to your local AI chatbot!")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": user_input}])
        print("AI: " + response['text'])

if __name__ == "__main__":
    chat()
