from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil
import os

# Yerelde ".env" dosyası varsa oradan okur; Render'da zaten panelden
# ayarlanan gerçek ortam değişkenleri kullanıldığı için bu satır orada etkisiz kalır.
load_dotenv()

import json
import random
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, List
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

    # Sohbet geçmişi tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            messages JSONB NOT NULL DEFAULT '[]',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)

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
bot_motoru = AIBot(model_name="openai/gpt-oss-120b", document_manager=kutuphaneci)

# --- VERİ ŞABLONLARI ---

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    user_id: str = "genel"

class ConversationSaveRequest(BaseModel):
    user_id: str
    title: str
    messages: List[dict]
    conversation_id: Optional[int] = None


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

# PDF/Belge/Görsel Yükleme Köprüsü
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form("genel")):
    DOCUMENT_EXTENSIONS = {
        ".pdf", ".docx", ".xlsx", ".txt",
        ".py", ".js", ".ts", ".html", ".css", ".java", ".cpp", ".c",
        ".cs", ".php", ".rb", ".go", ".rs", ".swift", ".kt", ".json",
        ".xml", ".yaml", ".yml", ".md", ".sh", ".bat"
    }
    ALLOWED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return {"status": "error", "message": f"Desteklenmeyen dosya formatı: {ext}. İzin verilenler: PDF, DOCX, XLSX, TXT ve JPG/PNG/WEBP/GIF gibi görseller."}

    print(f"Dosya alınıyor: {file.filename} (Sahibi: {user_id})")
    file_path = file.filename

    try:
        with open(file_path, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        if ext in IMAGE_EXTENSIONS:
            # Görsel: Groq'un vision modeliyle analiz et
            aciklama = bot_motoru.analyze_image(file_path)

            # Analiz sonucunu ileride "bu görselde ne vardı?" diye sorabilmek için hafızaya da ekle
            try:
                kutuphaneci.add_text_document(aciklama, filename=file.filename, user_id=user_id)
            except Exception as memory_error:
                print(f"Görsel hafızaya eklenemedi (analiz yine de döndürülüyor): {memory_error}")

            os.remove(file_path)
            return {"status": "success", "type": "image", "message": aciklama}

        else:
            kutuphaneci.add_document(file_path, user_id=user_id)
            os.remove(file_path)
            return {"status": "success", "type": "document", "message": f"'{file.filename}' başarıyla vektörel hafızaya eklendi!"}

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

# ==========================================
# --- 💬 SOHBET GEÇMİŞİ KÖPRÜLERİ ---
# ==========================================

# 1. Kullanıcının sohbet listesini getir (sidebar için, mesajlar hariç — hafif)
@app.get("/conversations")
def get_conversations(user_id: str):
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT id, title, updated_at FROM conversations
        WHERE user_id = %s ORDER BY updated_at DESC
    """, (user_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"status": "success", "conversations": rows}

# 2. Tek bir sohbetin tüm mesajlarını getir
@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int):
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT id, title, messages FROM conversations WHERE id = %s", (conversation_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if not row:
        return {"status": "error", "message": "Sohbet bulunamadı."}
    return {"status": "success", "conversation": row}

# 3. Sohbeti kaydet (yeni oluştur veya güncelle)
@app.post("/conversations/save")
def save_conversation(request: ConversationSaveRequest):
    conn = get_conn()
    cursor = conn.cursor()

    if request.conversation_id:
        cursor.execute("""
            UPDATE conversations SET messages = %s, updated_at = NOW()
            WHERE id = %s AND user_id = %s
        """, (json.dumps(request.messages), request.conversation_id, request.user_id))
        conn.commit()
        new_id = request.conversation_id
    else:
        cursor.execute("""
            INSERT INTO conversations (user_id, title, messages)
            VALUES (%s, %s, %s) RETURNING id
        """, (request.user_id, request.title, json.dumps(request.messages)))
        new_id = cursor.fetchone()[0]
        conn.commit()

    cursor.close()
    conn.close()
    return {"status": "success", "conversation_id": new_id}

# 4. Sohbeti sil
@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, user_id: str):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, user_id))
    conn.commit()
    cursor.close()
    conn.close()
    return {"status": "success", "message": "Sohbet silindi."}

# ==========================================
# --- 💻 GÜNLÜK YAZILIMCI SÖZLERİ ---
# ==========================================

YAZILIMCI_SOZLERI = [
    "Bugün o commit'i atacaksın!",
    "StackOverflow olmasaydı şu an nerede olurdun?",
    "'Çalışıyor ama neden çalıştığını bilmiyorum' da bir başarıdır.",
    "Kod yazmak kolaydır, kodu anlamak zordur — özellikle kendi kodun 6 ay sonra.",
    "Bug değil, dokümante edilmemiş özellik.",
    "Bir değişkene 'temp' adını verdiysen, o kalıcı olacak demektir.",
    "En iyi debug aracı: 'print' ve umut.",
    "Bugün derlenmezse, yarın senin sorunun olacak.",
    "'Sadece küçük bir değişiklik' dedin, 3 saat oldu.",
    "Git commit mesajın 'fix' ise, gerçekten neyi düzelttiğini sen de bilmiyorsun.",
    "Kahve bitti mi, sprint de biter.",
    "Production'da çalışıyorsa dokunma, nasıl çalıştığını sorma.",
    "İyi bir yazılımcı, iyi bir Google aramasıdır.",
    "'Bende çalışıyor' cümlesinin faturasını DevOps öder.",
    "Bugün yazdığın kod, yarının legacy code'u.",
    "Değişken isimlerine önem ver, gelecekteki sen sana teşekkür edecek.",
    "Recursion'ı anlamak için önce recursion'ı anlaman gerekir.",
    "Her 'TODO' yorumunun bir mezarlığı vardır.",
    "Bugün bir hata ayıkladın, yarın iki tane daha yaratacaksın.",
    "'Hızlıca bir hotfix' dedin, prod'u düşürdün.",
    "console.log() en sadık dostundur.",
    "Kod review'de sessiz kalmak onay değildir, korkaklıktır.",
    "Sen 'refactor edeceğim' dedikçe, teknik borç faiziyle büyür.",
    "İyi isimlendirilmiş bir fonksiyon, bin satır yorumdan iyidir.",
    "Bugün merge conflict'i çözersen, kahraman sensin.",
    "'Test yazacağım sonra' dediğin an, o 'sonra' hiç gelmez.",
    "Ctrl+Z, hayattaki en güvenilir arkadaşın.",
    "Bir API dokümantasyonu okumadan entegrasyona başlamak, gözü kapalı araba kullanmaktır.",
    "'Basit bir CRUD' dedin, 2 hafta oldu.",
    "Bugün kod yazdın, yarın o kodu lanetleyeceksin. Normal.",
    "Syntax hatası bulmak, treasure hunt'tan zordur bazen.",
    "İyi bir commit geçmişi, iyi bir günlük gibidir.",
    "'Sonra optimize ederim' cümlesi, en pahalı cümledir.",
    "Bir junior'a sabırla açıklamak, sen de öğrenmenin en iyi yoludur.",
    "Bugün her şey yolunda gidiyorsa, bir şeyi unutmuş olabilirsin."
]

@app.get("/motivasyon")
def get_motivasyon():
    # Her istekte listeyi karıştırıp bir kısmını dön — ticker'da tekrar tekrar çeşitlilik olsun
    secilenler = random.sample(YAZILIMCI_SOZLERI, k=min(12, len(YAZILIMCI_SOZLERI)))
    return {"status": "success", "sozler": secilenler}

# --- STATİK DOSYALAR VE ARAYÜZ ---

# avatar_logo.png gibi statik dosyaları serve et
app.mount("/static", StaticFiles(directory="."), name="static")