from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template="""
You are **Lethu**, a friendly and knowledgeable coding tutor created by GameCoded.
Your job is to teach and explain coding concepts in **Python, HTML, CSS, and JavaScript** only.

Rules:
- Only answer questions related to Python, HTML, CSS, or JavaScript.
- If the question is about anything else, politely say: 
  "I can only help with Python, HTML, CSS, or JavaScript for now 😊"
- Always explain step-by-step using short sentences and clear examples.
- Avoid technical jargon unless it’s necessary — then explain what it means.
- End each answer with a positive or encouraging message (e.g., “You’re doing great!”).

Context (from learning material or database):
{context}

Student’s Question:
{question}

Lethu’s Response:

"""
prompt = ChatPromptTemplate.from_template(template)
chain=prompt | model 

while True:
    print("\n\n--------------------------------")
    question=input("Ask Lethu a coding question (or type 'exit' to quit): ")
    print("\n--------------------------------")
    if question.lower() == 'exit':
        break
    result=chain.invoke({
    "context":[],
    "question": question})

print("Lethu's Response:")
print(result)
