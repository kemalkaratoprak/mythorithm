import os
from groq import Groq

class AIBot:
    def __init__(self, model_name="llama3-8b-8192", document_manager=None):
        self.model_name = model_name
        self.doc_manager = document_manager
        self.system_prompt = "Mythorithm adında uzman bir yazılım asistanısın. KESİNLİKLE SADECE TÜRKÇE konuşmalısın. Eğer sana ek bilgi (bağlam) verilirse, uydurmak yerine öncelikle o bilgiyi kullanarak cevap ver."
        self.history = []

        # Groq istemcisini başlatıyoruz (API key env'den okunur)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def reset_history(self):
        """Kısa süreli sohbet geçmişini sıfırlar (Yeni Sohbet için)"""
        self.history = []
        print("[PytHera] Sistem: Kısa süreli bot hafızası temizlendi. Yeni sohbete hazır.")

    def ask(self, prompt, temperature=0.7, user_id="genel"):
        user_message = prompt

        if self.doc_manager:
            results = self.doc_manager.search_memory(prompt, user_id=user_id)

            if results:
                context = "\n".join(results)
                print(f"\n[PytHera] {user_id} hafızasında ilgili belgeler bulundu, Groq'a iletiliyor...")

                user_message = f"""[SİSTEM İÇ BİLGİSİ]
Kullanıcının veritabanından gelen ek bilgiler (Bağlam):
{context}

[KULLANICI SORUSU]
{prompt}

[ÖNEMLİ KURALLAR]
1. Eğer soru ("nasılsın", "merhaba") gibi günlük bir sohbet ise, ek bilgileri TAMAMEN GÖRMEZDEN GEL ve sadece doğal bir insan gibi kısa cevap ver.
2. Eğer soru ek bilgilerle ilgiliyse, cevaplamak için onları kullan ama ASLA "Bana verdiğin bilgilere göre" gibi robotik cümleler kurma.
"""

        self.history.append({'role': 'user', 'content': user_message})

        # System prompt'u her seferinde başa ekliyoruz
        messages_to_send = [{'role': 'system', 'content': self.system_prompt}] + self.history

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_to_send,
                temperature=temperature,
                max_tokens=1024
            )
            bot_reply = response.choices[0].message.content
            self.history.append({'role': 'assistant', 'content': bot_reply})
            return bot_reply

        except Exception as e:
            return f"Bir hata oluştu: {e}"