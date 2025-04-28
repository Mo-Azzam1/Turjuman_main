# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import os
from dotenv import load_dotenv
import json_repair
from langchain_google_genai import ChatGoogleGenerativeAI # استخدام المكتبة الصحيحة لـ Gemini مع Langchain
import json # لإضافة الـ schema في الـ prompt


# --- Pydantic Models ---

# نموذج لمدخلات الـ API
class TranslationInput(BaseModel):
    word: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The English word to translate."
    )
    context: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The context paragraph where the word appears."
    )


# نموذج لمخرجات الـ API (نفس النموذج اللي عرفته)
class Translation(BaseModel):
    translated_word: str = Field(
        ...,
        min_length=1, # يمكن تكون كلمة واحدة
        max_length=255,
        description="The translated word in the target language (Arabic), based on the provided context."
    )
    target_synonyms: List[str] = Field(
        ...,
        min_items=0, # ممكن تكون القائمة فاضية لو النموذج معرفش يجيب مرادفات
        max_items=5,
        description="Different synonymous words in Arabic, relevant to the context. No duplicates."
    )
    source_synonyms: List[str] = Field(
        ...,
        min_items=0, # ممكن تكون القائمة فاضية
        max_items=5,
        description="Synonyms of the word in English. No duplicates."
    )
    definition: str = Field(
        ...,
        min_length=5,
        max_length=500, # زودت الحد الأقصى للتعريف شوية
        description="Definition of the original word in English."
    )
    example_usage: str = Field(
        ...,
        min_length=5,
        max_length=500, # زودت الحد الأقصى للمثال شوية
        description="An example sentence or phrase using the word to demonstrate its usage in context in English."
    )


# --- Helper Functions ---

def parse_json(text):
    """Attempts to parse text as JSON, using json_repair as a fallback."""
    try:
        # First, try standard JSON parsing (more strict)
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # If standard parsing fails, try json_repair
        try:
            return json_repair.loads(text)
        except Exception as e:
            print(f"json_repair failed: {e}") # طباعة الخطأ للمساعدة في Debug
            return None

def build_translation_messages(word: str, context: str) -> List[dict]:
    """Builds the message list for the Gemini API call."""
    # استخدام model_json_schema() في Pydantic V2 للحصول على الـ schema
    schema_json_string = json.dumps(Translation.model_json_schema(), ensure_ascii=False, indent=2)

    messages = [
        {
            "role": "system",
            "content": "\n".join([
                "You are a professional translator from English to Arabic.",
                "You will be provided with an English word and its context.",
                "Translate the word based on the context.",
                "Provide:",
                "- The translated word in Arabic.",
                "- A list of up to 5 relevant synonyms in Arabic (target language).",
                "- A list of up to 5 relevant synonyms in English (source language).",
                "- The English definition of the original word.",
                "- An example sentence using the original word in English.",
                "Your output must be a valid JSON object exactly matching the following Pydantic schema:",
                f"```json\n{schema_json_string}\n```", # تضمين الـ schema في الـ prompt
                "Do not add any extra text before or after the JSON.",
            ])
        },
        {
            "role": "user",
            "content": f"Context: {context.strip()}\nWord: {word.strip()}"
        }
    ]
    return messages

# --- Environment Setup & Model Initialization ---

# تحميل متغيرات البيئة من ملف .env (لو موجود)
load_dotenv()

# الحصول على مفتاح API من متغير البيئة
# استخدم نفس الاسم اللي استخدمته في الـ script الأول
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    # لو المفتاح مش موجود في متغيرات البيئة، ارفع خطأ
    # Railway هيتعرف على هذا الخطأ وهيفهم أن فيه مشكلة في التهيئة
    print("Error: GEMINI_API_KEY environment variable is not set.")
    print("Please set the GEMINI_API_KEY environment variable in your Railway project settings.")
    # يمكن استخدام sys.exit(1) هنا لو عايز التطبيق يفشل بسرعة لو المتغير مش موجود
    # لكن رفع Exception هو الطريقة القياسية في تهيئة التطبيقات
    raise ValueError("GEMINI_API_KEY environment variable is missing.")

# تهيئة نموذج Gemini (هتتم مرة واحدة عند بدء تشغيل التطبيق)
try:
    gemini_model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.5,
        google_api_key=gemini_api_key # استخدام المفتاح من متغير البيئة
    )
    print("Gemini model initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    # لو فشلت التهيئة هنا، التطبيق مش هيشتغل
    raise

# --- FastAPI App Setup ---

app = FastAPI(
    title="English to Arabic Translator API",
    description="API to translate an English word based on context using Gemini 1.5 Flash.",
    version="1.0.0"
)

# إضافة CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # اسمح بكل المواقع (يمكن تغييرها لنطاقات محددة لاحقًا)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- FastAPI Endpoints ---

@app.get("/")
async def read_root():
    """Root endpoint returning basic API info."""
    return {"message": "Translator API using Gemini 1.5 Flash is running.", "version": "1.0.0"}

@app.post("/translate", response_model=Translation)
async def translate_word_endpoint(input_data: TranslationInput):
    """
    Translates an English word to Arabic based on context using the Gemini 1.5 Flash model.

    Expects a JSON body with 'word' and 'context'.
    Returns a JSON object with translation details, synonyms, definition, and example usage.
    """
    messages = build_translation_messages(input_data.word, input_data.context)

    try:
        # استدعاء نموذج Gemini
        gemini_response = gemini_model.invoke(messages)

        # الحصول على محتوى الرد وتنظيفه من المسافات البيضاء الزائدة
        raw_json_string = gemini_response.content.strip()

        # محاولة تحليل JSON (مع إصلاح الأخطاء المحتملة)
        parsed_data = parse_json(raw_json_string)

        # التحقق مما إذا كان التحليل ناجحًا
        if parsed_data is None:
            print(f"Failed to parse or repair JSON for input: word='{input_data.word}', context='{input_data.context[:50]}...'")
            print(f"Raw response content: {raw_json_string}") # اطبع الرد الخام للمراجعة
            raise HTTPException(
                status_code=500,
                detail="Failed to parse or repair JSON response from the language model."
            )

        # التحقق من صحة البيانات المحللة باستخدام نموذج Pydantic
        # هذا سيضمن أن البيانات تتطابق مع الهيكل المتوقع وأنواع البيانات
        validated_data = Translation(**parsed_data)

        # إرجاع البيانات الصحيحة
        return validated_data

    except ValidationError as e:
        # Catch Pydantic validation errors if the parsed JSON doesn't match the model
        print(f"Pydantic validation error: {e.errors()}")
        print(f"Problematic JSON data: {parsed_data}") # اطبع البيانات اللي سببت الخطأ
        raise HTTPException(
            status_code=500,
            detail=f"Language model returned data that doesn't match the expected structure: {e.errors()}"
        )
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred during translation: {e}"
        )

# ملاحظة: للتشغيل المحلي، ستحتاج إلى تثبيت uvicorn وتشغيل الأمر:
# uvicorn main:app --reload
# للتوزيع على Railway، ستحتاج فقط لملف main.py وملف requirements.txt ومتغير البيئة GEMINI_API_KEY
