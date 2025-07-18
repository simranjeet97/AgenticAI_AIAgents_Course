# agent.py
import fitz  # PyMuPDF
from google.generativeai import GenerativeModel
from google.adk.agents import Agent

import os
os.environ["GOOGLE_API_KEY"] = "" # <--- REPLACE

def answer_question_from_pdf(pdf_bytes: bytes, question: str) -> dict:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        doc.close()

        model = GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Context:\n{text}\n\nQuestion:\n{question}")
        return {"status": "success", "answer": response.text.strip()}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

root_agent = Agent(
    name="pdf_agent",
    model="gemini-1.5-flash",
    description="Agent that answers questions based on uploaded PDFs.",
    instruction="""
You are an intelligent assistant that helps users understand PDF documents.

Your job is to:
- Extract and read the contents of PDF files.
- Answer user questions clearly and accurately using the content of the document.
- If a question cannot be answered from the document, respond with: "I couldn't find that information in the PDF."
- Format answers as follows:
  - Use **bold** for key terms or names.
  - Use bullet points for lists.
  - Use numbered steps for procedures.
  - Use markdown code blocks (```text```) for code or quoted sections.

Be concise, use plain language, and structure answers clearly.
""",
    tools=[answer_question_from_pdf],
)