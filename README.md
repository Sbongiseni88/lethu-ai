🧠 Lethu AI Chatbot — GameCoded Learning Agent

Lethu is an AI-powered chatbot designed to teach kids (ages 8–14) the basics of Python, JavaScript, HTML, and CSS in a fun, gamified way 🎮✨

This repo contains:

The Lethu AI agent prompt templates

Scripts for refining or fine-tuning lightweight open models (like TinyLlama)

A small dataset for RAG (Retrieval-Augmented Generation)

Code for integrating the agent into the GameCoded Android app (Kotlin frontend + Python AI backend)

🚀 Project Goal

To create an offline-first AI chatbot that:

Runs locally on Android using TinyLlama (GGUF) via Ollama or llama.cpp

Switches to a cloud-hosted model (via Firebase backend or vLLM API) when connected

Helps kids learn coding step-by-step with hints, quizzes, and fun motivational feedback

🧩 Core Features
Feature	Description
🎓 AI Tutor	Teaches coding concepts interactively with explanations, hints, and quizzes.
📚 RAG (Retrieval-Augmented Generation)	Uses a local CSV dataset to help Lethu recall code examples and definitions.
🌐 Language Flexibility	Can translate explanations (not code) into any of South Africa’s 11 official languages.
🔒 Offline Mode	Runs completely offline using TinyLlama, MLC, or Ollama local inference.
☁️ Online Mode	Connects to a remote model (like Llama 3 or Mistral) for richer responses when online.
🧠 Lesson Context Tracking	Maintains lesson progress, history, and motivational feedback.
