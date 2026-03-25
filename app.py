import os
import re
import sys
print("PYTHON USED:", sys.executable)
import sys
print(sys.executable)
import faiss
import PyPDF2
from flask import Flask, render_template, request

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



openai_api_key = os.getenv("OPENROUTER_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENROUTER_API_KEY")

print("FAISS loaded:", faiss.__version__)

app = Flask(__name__)


def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def clean_text(raw_text):
    cleaned = re.sub(r'RGEE.*?page \d+ /?\d*', '', raw_text)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    cleaned = re.sub(r'[^\x00-\x7F]+',' ', cleaned)
    # Supprime les doubles espaces
    cleaned = re.sub(r'\s+',' ', cleaned)
    return cleaned.strip()


def chunk_text(text):
    chunks = []
    chapters = re.split(r'(CHAPITRE \d+)', text)
    current_chapter = ""

    for part in chapters:
        if re.match(r'CHAPITRE \d+', part):
            current_chapter = part.strip()
        else:
            sections = re.split(r'(\d+\.\d+(?:\.\d+)*)', part)
            section_title = ""
            section_text = ""

            for s in sections:
                if not s:
                    continue
                if re.match(r'\d+\.\d+(\.\d+)*', s):
                    if section_text:
                        chunks.append({
                            "chapter": current_chapter,
                            "section": section_title,
                            "content": section_text.strip()
                        })
                    section_title = s.strip()
                    section_text = ""
                else:
                    section_text += s.strip() + "\n"

            if section_text:
                chunks.append({
                    "chapter": current_chapter,
                    "section": section_title,
                    "content": section_text.strip()
                })
    return chunks




pdf_file = "data/University Luis and Pasteur Reglementations.pdf"

raw_text = extract_text(pdf_file)
cleaned_text = clean_text(raw_text)
chunks = chunk_text(cleaned_text)

documents = [
    Document(
        page_content=c["content"],
        metadata={
            "chapter": c["chapter"],
            "section": c["section"]
        }
    )
    for c in chunks
]

print("Total documents:", len(documents))

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
)

vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="openai/gpt-4o-mini",
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_template("""
You are an assistant answering questions about university regulations.

Use ONLY the context provided below.
If the answer is not found in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

parser = StrOutputParser()



def ask_question(question):
    docs = retriever.invoke(question)

    context = "\n\n".join([
        f"Chapter: {d.metadata.get('chapter','')}\n"
        f"Section: {d.metadata.get('section','')}\n"
        f"{d.page_content}"
        for d in docs
    ])

    chain = prompt | llm | parser

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer




from flask import jsonify

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"]
    answer = ask_question(question)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)