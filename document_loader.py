import os
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

class DocumentManager:
    def __init__(self):
        """Pinecone bağlantısını kurar. API key environment variable'dan okunur."""
        
        # Embedding modeli (metni vektöre çeviren motor)
        print("Sistem: Embedding modeli yükleniyor...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Pinecone bağlantısı
        api_key = os.environ.get("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        
        index_name = "mythority-memory"
        
        # Index yoksa otomatik oluştur
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"Sistem: '{index_name}' index'i bulunamadı, oluşturuluyor...")
            self.pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 modeli 384 boyutlu vektör üretir
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Sistem: '{index_name}' index'i başarıyla oluşturuldu.")
        
        self.index = self.pc.Index(index_name)
        print("Sistem: Pinecone vektör veritabanı aktif.\n")

    def add_document(self, file_path, user_id):
        """PDF dosyasını okur, parçalar, vektöre çevirir ve Pinecone'a kullanıcı etiketiyle ekler."""
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(pages)

        vectors = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            embedding = self.embedder.encode(text).tolist()
            doc_id = f"{user_id}_{os.path.basename(file_path)}_chunk_{i}"

            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {
                    "owner": user_id,
                    "text": text,
                    "filename": os.path.basename(file_path)
                }
            })

        # Pinecone'a toplu olarak gönderiyoruz (100'lük paketler halinde)
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
                filter={"owner": {"$eq": user_id}},  # Sadece bu kullanıcının belgeleri
                include_metadata=True
            )

            texts = [match["metadata"]["text"] for match in results["matches"]]
            return texts

        except Exception as e:
            print(f"Arama hatası: {e}")
            return []

    def get_all_documents(self, user_id):
        """Kullanıcıya ait benzersiz dosya isimlerini getirir."""
        try:
            # Pinecone'da metadata filtresiyle listeleme yapıyoruz
            results = self.index.query(
                vector=[0.0] * 384,  # Boş vektör, sadece metadata için
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