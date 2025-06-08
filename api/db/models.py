from sqlalchemy import Column, Integer, String, Float
from .base import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    prob_neg = Column(Float)
    prob_neu = Column(Float)
    prob_pos = Column(Float)
