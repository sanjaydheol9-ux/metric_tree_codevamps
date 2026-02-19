from sqlalchemy import Column, Integer, Float
from database import Base

class WeeklyMetrics(Base):
    __tablename__ = "weekly_metrics1"

    id = Column(Integer, primary_key=True, index=True)
    week = Column(Integer, nullable=False)
    delivery_score = Column(Float)
    accuracy_score = Column(Float)
    dispatch_score = Column(Float)
    warehouse_score = Column(Float)
    on_time_score = Column(Float)
