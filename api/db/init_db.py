from .db import engine
from .base import Base
from .models import Prediction

# Tüm tabloları oluşturur
Base.metadata.create_all(bind=engine)
