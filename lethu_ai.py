from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, get_retriever

model = OllamaLLM(model="llama3.2")

# Toggle debug output
DEBUG = False

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

# --- Friendly intro and language selection ---
print("Hello — I'm Lethu, your friendly coding tutor! I can help with Python, HTML, CSS, or JavaScript.")
print("Before we start, which language would you like to focus on? (python/html/css/javascript)")
supported = {"python", "html", "css", "javascript"}
chosen_language = ""
while True:
    chosen_language = input("Choose language: ").strip().lower()
    if chosen_language in supported:
        break
    print("Please choose one of: python, html, css, javascript")

# Use a smaller k for faster retrieval (lower latency). You can increase k later if you want more context.
# Lower k to 1 for minimal retrieval latency; increase if you want more context returned.
retriever = get_retriever(k=1)

while True:
    print("\n\n--------------------------------")
    question=input("Ask Lethu a coding question (or type 'exit' to quit): ")
    print("\n--------------------------------")
    if question.lower() == 'exit':
        break
    # Provide immediate feedback to the user so the UI doesn't feel frozen
    

    # Accept multiple retriever interfaces: invoke(), get_relevant_documents(), callable returning list
    content = None
    try:
        # common pattern in our vector.py: retriever.invoke(question) -> list[Document]
        if hasattr(retriever, "invoke"):
            content = retriever.invoke(question)
    except Exception as e:
        print("[retriever.invoke error]", e)

    if content is None and hasattr(retriever, "get_relevant_documents"):
        try:
            content = retriever.get_relevant_documents(question)
        except Exception as e:
            print("[retriever.get_relevant_documents error]", e)

    if content is None and callable(retriever):
        try:
            content = retriever(question)
        except Exception as e:
            print("[retriever(...) error]", e)

    # If retriever returned a list of Document-like objects, join their page_content
    if isinstance(content, (list, tuple)):
        # Filter by chosen language when possible (metadata 'source' used in vector.py)
        filtered = []
        for d in content:
            meta = d.metadata if hasattr(d, 'metadata') else (d.get('metadata') if isinstance(d, dict) else {})
            src = None
            if isinstance(meta, dict):
                src = meta.get('source') or meta.get('lang')
            # accept if no metadata or matches chosen language
            if not src or src.strip().lower() == chosen_language:
                filtered.append(d)

        parts = []
        for d in filtered:
            if isinstance(d, dict):
                parts.append(d.get("page_content") or d.get("content") or str(d))
            else:
                parts.append(getattr(d, "page_content", None) or getattr(d, "content", None) or str(d))
        context_text = "\n\n".join([p for p in parts if p])
    else:
        context_text = content or ""

  

    # Debug: show retrieved content when DEBUG is enabled
    if DEBUG:
        print("DEBUG: retrieved content (repr):", repr(content))
        print("DEBUG: retrieved content type:", type(content))

    # Debug: show retrieved content when DEBUG is enabled
    if DEBUG:
        print("DEBUG: retrieved content (repr):", repr(content))
        print("DEBUG: retrieved content type:", type(content))

    # Try to stream the model's output when possible for faster perceived responses
    prompt_text = template.format(context=context_text, question=question)
    streamed = False

    # Attempt model.stream(prompt_text)
    try:
        if hasattr(model, "stream") and callable(getattr(model, "stream")):
            
            for chunk in model.stream(prompt_text):
                if isinstance(chunk, dict):
                    text = chunk.get("text") or chunk.get("content") or str(chunk)
                else:
                    text = str(chunk)
                print(text, end="", flush=True)
            print("\n")
            streamed = True
    except Exception as e:
        if DEBUG:
            print("[model.stream error]", e)

    # Attempt model.generate(..., stream=True) if available
    if not streamed:
        try:
            if hasattr(model, "generate"):
                gen = None
                try:
                    gen = model.generate(prompt_text, stream=True)
                except TypeError:
                    gen = None

                if gen is not None and hasattr(gen, "__iter__"):
                    
                    for chunk in gen:
                        if isinstance(chunk, dict):
                            text = chunk.get("text") or chunk.get("content") or str(chunk)
                        else:
                            text = str(chunk)
                        print(text, end="", flush=True)
                    print("\n")
                    streamed = True
        except Exception as e:
            if DEBUG:
                print("[model.generate streaming error]", e)

    # Helper: decorate output with emojis based on chosen language
    def decorate_with_emojis(text: str, lang: str) -> str:
        lang_emoji = {
            "python": "🐍",
            "javascript": "✨",
            "html": "📄",
            "css": "🎨"
        }.get(lang, "💡")
        # Add a friendly ending emoji if not present
        ending = " ✅"
        # Keep it short: prefix with language emoji and suffix with ending
        return f"{lang_emoji} {text.strip()} {ending}"

    # If we streamed the response, skip the non-streaming path
    if streamed:
        # For streaming we can't decorate the whole text easily, so just print a small emoji at the end
        print("\n🙂\n")
        continue

    # Fallback: call the chain or model normally
    result = None
    try:
        if hasattr(chain, "invoke"):
            result = chain.invoke({"context": context_text, "question": question})
        else:
            result = chain({"context": context_text, "question": question})
    except Exception as e:
        print("[chain.invoke error]", e)
        # Try calling the model directly with the formatted prompt
        try:
            if hasattr(model, "invoke"):
                result = model.invoke(prompt_text)
            elif hasattr(model, "generate"):
                result = model.generate(prompt_text)
            else:
                result = str(prompt_text)
        except Exception as e2:
            print("[model direct call error]", e2)
            result = None

    # Debug: show raw result when DEBUG is enabled
    if DEBUG:
        print("DEBUG: raw result (repr):", repr(result))
        print("DEBUG: result type:", type(result))

    # Extract string output from result
    output = None
    if isinstance(result, str):
        output = result
    elif isinstance(result, dict):
        output = result.get("text") or result.get("output_text") or result.get("response")
    else:
        output = getattr(result, "text", None) or getattr(result, "output_text", None) or getattr(result, "content", None)

    if not output:
        print("Lethu couldn't produce an answer. Check the DEBUG lines above.")
    else:
        decorated = decorate_with_emojis(output, chosen_language)
        print("Lethu's Answer:\n")
        print(decorated)



 
