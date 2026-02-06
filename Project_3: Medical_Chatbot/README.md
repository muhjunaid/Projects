#  AI-DOCTOR --- Medical Chatbot (Analyze Image & Text)

AI-DOCTOR is an intelligent medical chatbot web application that
analyzes *medical images* along with *user-provided symptoms text*
to generate AI-based medical guidance.

The system uses *FastAPI* as backend and integrates the **Meta Llama-4
Scout 17B Instruct model** through the *GROQ API* to perform
multimodal understanding (Image + Text).

------------------------------------------------------------------------

## Features

-   Upload medical image (skin issue, wound, rash, report, etc.)
-   Enter symptoms in text form
-   Multimodal analysis (Image + Text together)
-   AI-generated medical suggestions
-   FastAPI powered backend
-   Clean web interface using Jinja2 templates
-   Real-time response using GROQ API

------------------------------------------------------------------------

## Technologies Used

  Technology               Purpose
  ------------------------ -----------------------
  FastAPI                  Backend framework
  Jinja2 Templates         Frontend rendering
  StaticFiles              Serving static assets
  PIL                      Image processing
  Base64 Encoding          Image transmission
  GROQ API                 LLM access
  Meta Llama-4 Scout 17B   AI reasoning
  HTML/CSS                 User Interface
  Python Dotenv            Environment variables

------------------------------------------------------------------------

## How It Works

1.  User uploads a medical image.
2.  User enters symptoms in text box.
3.  Image is converted into Base64 format.
4.  FastAPI sends image + text to GROQ API.
5.  Llama-4 model analyzes both inputs.
6.  AI generates medical guidance.
7.  Result is displayed on the web page.

------------------------------------------------------------------------

## Installation


python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Create a `.env` file:

    GROQ_API_KEY=your_api_key_here

Run:

python file: app.py

Open: http://127.0.0.1:8000
