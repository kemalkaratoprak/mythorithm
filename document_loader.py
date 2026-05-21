import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

class DocumentManager:
    def __init__(self):
        """Pinecone bağlantısını kurar. API key environment variable'dan okunur."""
        print("Sistem: Embedding modeli yükleniyor...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        api_key = os.environ.get("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)

        index_name = "mythority-memory"

        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"Sistem: '{index_name}' index'i bulunamadı, oluşturuluyor...")
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Sistem: '{index_name}' index'i başarıyla oluşturuldu.")

        self.index = self.pc.Index(index_name)
        print("Sistem: Pinecone vektör veritabanı aktif.\n")

    def read_file(self, file_path):
        """Dosya uzantısına göre doğru okuyucuyu seçer ve metni döndürür."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            from pypdf import PdfReader
            text = ""
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text

        elif ext == ".docx":
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            # Tablolardaki metni de okuyalım
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text += "\n" + cell.text.strip()
            return text

        elif ext == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            text = ""
            for sheet in wb.worksheets:
                text += f"[Sayfa: {sheet.title}]\n"
                for row in sheet.iter_rows(values_only=True):
                    row_text = "\t".join([str(c) for c in row if c is not None])
                    if row_text.strip():
                        text += row_text + "\n"
            return text

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext in {".py", ".js", ".ts", ".html", ".css", ".java", ".cpp", ".c",
                     ".cs", ".php", ".rb", ".go", ".rs", ".swift", ".kt", ".json",
                     ".xml", ".yaml", ".yml", ".md", ".sh", ".bat"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Dosya adını ve dilini bağlama ekle ki bot daha iyi anlasın
            return f"[Dosya: {os.path.basename(file_path)}]\n\n{content}"

        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")

    def add_document(self, file_path, user_id):
        """Dosyayı okur, parçalar, vektöre çevirir ve Pinecone'a kullanıcı etiketiyle ekler."""
        text = self.read_file(file_path)

        if not text.strip():
            raise ValueError("Dosyadan metin okunamadı veya dosya boş.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedder.encode(chunk).tolist()
            doc_id = f"{user_id}_{os.path.basename(file_path)}_chunk_{i}"
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {
                    "owner": user_id,
                    "text": chunk,
                    "filename": os.path.basename(file_path)
                }
            })

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i + batch_size])

        print(f"Sistem: {os.path.basename(file_path)} dosyası {user_id} hafızasına kaydedildi.")

    def search_memory(self, query, user_id, n_results=3):
        """Sadece ilgili kullanıcıya ait belgeler arasında arama yapar."""
        try:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.index.query(
                vector=query_embedding,
                top_k=n_results,
                filter={"owner": {"$eq": user_id}},
                include_metadata=True
            )
            return [match["metadata"]["text"] for match in results["matches"]]
        except Exception as e:
            print(f"Arama hatası: {e}")
            return []

    def get_all_documents(self, user_id):
        """Kullanıcıya ait benzersiz dosya isimlerini getirir."""
        try:
            results = self.index.query(
                vector=[0.0] * 384,
                top_k=1000,
                filter={"owner": {"$eq": user_id}},
                include_metadata=True
            )
            unique_docs = set()
            for match in results["matches"]:
                filename = match["metadata"].get("filename", "")
                if filename:
                    unique_docs.add(filename)
            return list(unique_docs)
        except Exception as e:
            print(f"Hafıza okuma hatası: {e}")
            return []

    def clear_memory(self):
        """Tüm vektörel hafızayı sıfırlar."""
        try:
            self.index.delete(delete_all=True)
            print("Sistem: Pinecone hafızası tamamen sıfırlandı.")
            return True
        except Exception as e:
            print(f"Sıfırlama hatası: {e}")
            return False