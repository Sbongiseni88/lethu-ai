from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os
import csv

# Read rag.csv robustly. The provided CSV has some malformed rows/extra quotes and inconsistent
# quoting. First try pandas with the python engine and liberal parsing, otherwise fall back to
# a safe csv.reader pass that builds rows manually.
csv_path = "rag.csv"
try:
    # engine='python' is more tolerant of irregular quoting/field counts
    df = pd.read_csv(csv_path, engine="python", skip_blank_lines=True)
except Exception:
    # Fallback: read with the stdlib csv module and normalize rows to 4 columns
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            # Skip empty rows
            if not any(cell.strip() for cell in r):
                continue
            # If row has more than 4 columns, join the extras into the last column
            if len(r) > 4:
                r = r[:3] + [" ".join(r[3:])]
            # If row has fewer than 4 columns, pad with empty strings
            if len(r) < 4:
                r = r + [""] * (4 - len(r))
            rows.append(r)
    df = pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame(columns=["Language", "Concept", "Description", "Example"])
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Build content from the CSV columns present in rag.csv
        content = f"{row.get('Concept','')} {row.get('Description','')} {row.get('Example','')}"
        doc = Document(
            page_content=content,
            metadata={"source": row.get("Language", "")},
            id=str(i)
        )
        documents.append(doc)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="rag_collection",
    persist_directory=db_location,
    embedding_function=embeddings
)
try:
    if add_documents:
        # documents and ids only exist when add_documents is True
        vector_store.add_documents(documents=documents, ids=ids)

    # default k used by callers who import `retriever` directly
    DEFAULT_RETRIEVER_K = 3
    def get_retriever(k: int | None = None):
        """Return a retriever configured with search k (smaller k -> faster retrieval)."""
        use_k = DEFAULT_RETRIEVER_K if k is None else k
        return vector_store.as_retriever(search_kwargs={"k": use_k})

    retriever = get_retriever()
except Exception as e:
    # If anything goes wrong (missing Chroma, bad embeddings, etc.), provide a fallback
    # retriever with a minimal interface used by the main script. The fallback returns
    # an empty context so the model can still answer without RAG context.
    print("[vector.py] Warning: vector store unavailable, using fallback retriever:", e)

    class FallbackRetriever:
        def invoke(self, q):
            return ""

        def get_relevant_documents(self, q):
            return []

        def __call__(self, q):
            return ""

    retriever = FallbackRetriever()
    def get_retriever(k: int | None = None):
        return retriever
