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

# Difficulty selection (Phase A)
print("What difficulty level would you like? (beginner/intermediate/advanced)")
supported_levels = {"beginner", "intermediate", "advanced"}
level = "beginner"
while True:
    level = input("Choose level: ").strip().lower()
    if level in supported_levels:
        break
    print("Please choose: beginner, intermediate, or advanced")

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

        # --- Phase A follow-up prompt ---
        print("\nWould you like an example, a quick quiz, more detail, or skip? (example/quiz/more/skip)")
        follow = input("Your choice: ").strip().lower()

        # Simple fuzzy matching
        if follow.startswith("e"):
            # Ask for an example from the model: short code snippet or demonstration
            try:
                prompt_example = f"Provide a short, clear example for this concept in {chosen_language} at {level} level. Context: {context_text} Question: {question}"
                example_res = None
                if hasattr(model, "invoke"):
                    example_res = model.invoke(prompt_example)
                elif hasattr(model, "generate"):
                    example_res = model.generate(prompt_example)
                example_text = example_res if isinstance(example_res, str) else (example_res.get('text') if isinstance(example_res, dict) else str(example_res))
                print("\nExample:\n")
                print(decorate_with_emojis(example_text or "(no example available)", chosen_language))
            except Exception as e:
                print("Could not generate example:", e)

        elif follow.startswith("q"):
            # Quick quiz: ask the model to generate 1 multiple-choice question and evaluate the answer
            try:
                quiz_prompt = f"Create a 1-question multiple choice quiz (A/B/C) about: {question}. Include the correct answer labeled. Keep it short."
                quiz_res = None
                if hasattr(model, "invoke"):
                    quiz_res = model.invoke(quiz_prompt)
                elif hasattr(model, "generate"):
                    quiz_res = model.generate(quiz_prompt)
                quiz_text = quiz_res if isinstance(quiz_res, str) else (quiz_res.get('text') if isinstance(quiz_res, dict) else str(quiz_res))
                print("\nQuick Quiz:\n")
                print(quiz_text)
                user_ans = input("Your answer (A/B/C): ").strip().upper()
                # Ask the model to grade the answer
                grade_prompt = f"The quiz was: {quiz_text}\nUser answered: {user_ans}. Is this correct? Reply only 'correct' or 'incorrect' and one short sentence of feedback."
                grade_res = None
                if hasattr(model, "invoke"):
                    grade_res = model.invoke(grade_prompt)
                elif hasattr(model, "generate"):
                    grade_res = model.generate(grade_prompt)
                grade_text = grade_res if isinstance(grade_res, str) else (grade_res.get('text') if isinstance(grade_res, dict) else str(grade_res))
                print("\nResult:\n", grade_text)
            except Exception as e:
                print("Could not generate quiz:", e)

        elif follow.startswith("m"):
            # More detail: ask the model for a deeper explanation
            try:
                more_prompt = f"Give a deeper, step-by-step explanation for: {question} in {chosen_language} at {level} level. Context: {context_text}"
                more_res = None
                if hasattr(model, "invoke"):
                    more_res = model.invoke(more_prompt)
                elif hasattr(model, "generate"):
                    more_res = model.generate(more_prompt)
                more_text = more_res if isinstance(more_res, str) else (more_res.get('text') if isinstance(more_res, dict) else str(more_res))
                print("\nMore detail:\n")
                print(decorate_with_emojis(more_text or "(no additional detail)", chosen_language))
            except Exception as e:
                print("Could not generate more detail:", e)

        else:
            # skip or unknown input - continue the loop
            continue



 
