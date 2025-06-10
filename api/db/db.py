import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# .env'den DATABASE_URL değişkenini oku, yoksa varsayılan değeri kullan
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/duygulu")

# Engine ve session ayarları
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

# Bağlantı yönetimi
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
