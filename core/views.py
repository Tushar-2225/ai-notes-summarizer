from django.shortcuts import render
from django.http import HttpResponse
from transformers import pipeline
import nltk
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Download tokenizer (once)
nltk.download('punkt', quiet=True)

# --- Global caches ---
summarizer = None
qg = None


def get_summarizer():
    """Use a small summarization model for Render (fast + low memory)."""
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    return summarizer


def get_qg():
    """Use a lightweight T5 model for question generation."""
    global qg
    if qg is None:
        qg = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")
    return qg


def home(request):
    summary = None
    flashcards = []

    if request.method == "POST":
        text = request.POST.get("text")

        if "summarize" in request.POST:
            if text:
                summarizer_model = get_summarizer()
                summarized = summarizer_model(text, max_length=120, min_length=25, do_sample=False)
                summary = summarized[0]["summary_text"]
            else:
                summary = "Please enter some text to summarize."

        elif "flashcards" in request.POST:
            summary_text = request.POST.get("summary")
            if summary_text:
                qg_model = get_qg()
                sentences = nltk.sent_tokenize(summary_text)
                for sent in sentences:
                    qa = qg_model(f"generate question: {sent}")
                    flashcards.append({
                        "question": qa[0]["generated_text"],
                        "answer": sent
                    })

        elif "download_pdf" in request.POST:
            questions = request.POST.getlist("questions[]")
            answers = request.POST.getlist("answers[]")

            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.setFont("Helvetica", 12)

            y = 750
            p.drawString(100, y, "AI-Generated Flashcards")
            y -= 30

            for i, (q, a) in enumerate(zip(questions, answers), start=1):
                p.drawString(80, y, f"Q{i}: {q}")
                y -= 20
                p.drawString(100, y, f"A{i}: {a}")
                y -= 30

                if y < 100:
                    p.showPage()
                    p.setFont("Helvetica", 12)
                    y = 750

            p.save()
            buffer.seek(0)

            response = HttpResponse(buffer, content_type="application/pdf")
            response["Content-Disposition"] = 'attachment; filename="flashcards.pdf"'
            return response

    return render(request, "core/home.html", {"summary": summary, "flashcards": flashcards})
