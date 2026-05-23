from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from ai_engine import AIBot
from document_loader import DocumentManager

app = FastAPI(title="Mythorithm API")


@app.get("/robots.txt", include_in_schema=False)
async def get_robots():
    return FileResponse("robots.txt")

@app.get("/sitemap.xml", include_in_schema=False)
async def get_sitemap():
    return FileResponse("sitemap.xml")

# --- POSTGRESQL VERİTABANI AYARLARI ---
def get_conn():
    """Railway'in otomatik sağladığı DATABASE_URL ile bağlantı kurar."""
    return psycopg2.connect(os.environ.get("DATABASE_URL"))

def veritabanini_hazirla():
    conn = get_conn()
    cursor = conn.cursor()
    # SQLite'daki AUTOINCREMENT → PostgreSQL'de SERIAL
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Başlangıç kullanıcıları — şifreler environment variable'dan okunuyor
    admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")
    test_kullanicilari = [("admin", admin_pass)]
    for user, target_pass in test_kullanicilari:
        cursor.execute("""
            INSERT INTO users (username, password)
            VALUES (%s, %s)
            ON CONFLICT (username) DO NOTHING
        """, (user, target_pass))

    conn.commit()
    cursor.close()
    conn.close()
    print("Sistem: PostgreSQL Veritabanı ve Kullanıcı Tablosu Kararlı.")

# Sunucu her açıldığında veritabanını hazırla
veritabanini_hazirla()

# Güvenlik Duvarı İzni
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Kütüphaneciyi Başlatıyoruz (Pinecone)
kutuphaneci = DocumentManager()

# 2. Motoru Başlatıyor ve Kütüphaneciyi Ona Bağlıyoruz
bot_motoru = AIBot(model_name="llama-3.3-70b-versatile", document_manager=kutuphaneci)

# --- VERİ ŞABLONLARI ---

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    user_id: str = "genel"


# --- KÖPRÜLER (ENDPOINTS) ---

# Ana sayfa
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/app")
def serve_frontend_app():
    return FileResponse("index.html")

@app.get("/status")
def read_root():
    return {"durum": "aktif", "sistem": "Mythorithm API Çalışıyor"}

# --- GİRİŞ YAP KÖPRÜSÜ ---
@app.post("/login")
def login_user(request: LoginRequest):
    username_clean = request.username.lower().strip()

    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = %s", (username_clean,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result and result[0] == request.password:
        print(f"Sistem: {username_clean} PostgreSQL doğrulmasıyla giriş yaptı.")
        return {"status": "success", "username": username_clean}
    else:
        print(f"Sistem: {username_clean} için hatalı giriş denemesi!")
        return {"status": "error", "message": "Kullanıcı adı veya şifre hatalı!"}


# --- KAYIT OL KÖPRÜSÜ ---
@app.post("/register")
def register_user(request: LoginRequest):
    username_clean = request.username.lower().strip()

    if not username_clean or not request.password.strip():
        return {"status": "error", "message": "Kullanıcı adı veya şifre boş bırakılamaz!"}

    conn = get_conn()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username_clean, request.password)
        )
        conn.commit()
        print(f"Sistem: Yeni kullanıcı PostgreSQL'e yazıldı -> {username_clean}")
        return {"status": "success", "message": "Kayıt işlemi başarıyla tamamlandı!"}
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        print(f"Sistem: {username_clean} adı zaten veritabanında var!")
        return {"status": "error", "message": "Bu kullanıcı adı zaten alınmış!"}
    finally:
        cursor.close()
        conn.close()

# Mesajlaşma Köprüsü
@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    print(f"Soru geldi: {request.message} (Kullanıcı: {request.user_id}, Yaratıcılık: {request.temperature})")
    cevap = bot_motoru.ask(request.message, temperature=request.temperature, user_id=request.user_id)
    return {"response": cevap}

# PDF Yükleme Köprüsü
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form("genel")):
    ALLOWED_EXTENSIONS = {
        ".pdf", ".docx", ".xlsx", ".txt",
        ".py", ".js", ".ts", ".html", ".css", ".java", ".cpp", ".c",
        ".cs", ".php", ".rb", ".go", ".rs", ".swift", ".kt", ".json",
        ".xml", ".yaml", ".yml", ".md", ".sh", ".bat"
    }
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return {"status": "error", "message": f"Desteklenmeyen dosya formatı: {ext}. İzin verilenler: PDF, DOCX, XLSX, TXT"}

    print(f"Dosya alınıyor: {file.filename} (Sahibi: {user_id})")
    file_path = file.filename

    try:
        with open(file_path, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        kutuphaneci.add_document(file_path, user_id=user_id)
        os.remove(file_path)
        return {"status": "success", "message": f"'{file.filename}' başarıyla vektörel hafızaya eklendi!"}

    except Exception as e:
        print(f"Yükleme sırasında hata: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"status": "error", "message": f"Hata oluştu: {str(e)}"}

# Hafızadaki Belgeleri Listeleme Köprüsü
@app.get("/documents")
def get_loaded_documents(user_id: str = "genel"):
    print(f"Sistem: {user_id} için hafıza listesi isteniyor.")
    docs = kutuphaneci.get_all_documents(user_id=user_id)
    return {"documents": docs}

# Hafızayı Sıfırlama Köprüsü
@app.post("/clear-memory")
def clear_vector_memory():
    success = kutuphaneci.clear_memory()
    if success:
        return {"status": "success", "message": "Hafıza başarıyla sıfırlandı."}
    return {"status": "error", "message": "Hafıza sıfırlanırken hata oluştu."}

# ==========================================
# --- 🛡️ ADMİN PANELİ KONTROL KÖPRÜLERİ ---
# ==========================================

# 1. Tüm Kullanıcıları Getir
@app.get("/admin/users")
def get_all_users():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()

    user_list = [{"id": u[0], "username": u[1]} for u in users]
    return {"status": "success", "users": user_list}

# 2. İstenilen Kullanıcıyı Sistemden Sil
@app.delete("/admin/users/{username_to_delete}")
def delete_user(username_to_delete: str):
    if username_to_delete.lower() == "admin":
        return {"status": "error", "message": "Ana yönetici (admin) hesabı silinemez!"}

    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = %s", (username_to_delete,))
    conn.commit()
    cursor.close()
    conn.close()

    print(f"Sistem Uyarısı: '{username_to_delete}' adlı kullanıcı sistemden atıldı.")
    return {"status": "success", "message": f"'{username_to_delete}' kullanıcısı sistemden silindi."}

# Yeni Sohbet için Hafıza Sıfırlama Köprüsü
@app.post("/reset-chat")
def reset_chat_history():
    bot_motoru.reset_history()
    return {"status": "success", "message": "Bot hafızası sıfırlandı."}

# --- STATİK DOSYALAR VE ARAYÜZ ---

# avatar_logo.png gibi statik dosyaları serve et
app.mount("/static", StaticFiles(directory="."), name="static")