import os
from datetime import datetime, timezone
import json
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

# Carrega variáveis do .env
load_dotenv(Path(__file__).resolve().parent / ".env")
DATABASE_URL = os.getenv("BD")

if not DATABASE_URL:
    raise ValueError("Variável de ambiente 'BD' não encontrada. Verifique seu arquivo .env.")

# O SQLAlchemy necessita que a string comece com postgresql:// (o NeonDB já fornece isso).
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Missao(Base):
    __tablename__ = "missoes"
    id = Column(Integer, primary_key=True, index=True)
    sessao = Column(String, index=True)
    inicio = Column(Float, nullable=False) # time.time()
    fim = Column(Float, nullable=True)
    duracao = Column(Float, nullable=True)
    status = Column(String, default="EM_ANDAMENTO") # EM_ANDAMENTO, CONCLUIDA
    alertas = Column(Integer, default=0)
    modelo = Column(String, nullable=True)

class EventoAlerta(Base):
    __tablename__ = "eventos_alerta"
    id = Column(Integer, primary_key=True, index=True)
    missao_id = Column(Integer, index=True) # Ligação lógica
    timestamp = Column(String, nullable=False) # ISO format
    nivel = Column(String, nullable=False)
    mensagem = Column(String, nullable=False)
    track_id = Column(Integer, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    alt = Column(Float, nullable=True)

class TelemetriaPonto(Base):
    __tablename__ = "telemetria_pontos"
    id = Column(Integer, primary_key=True, index=True)
    missao_id = Column(Integer, index=True)
    timestamp = Column(Float, nullable=False)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    alt = Column(Float, nullable=True)
    velocidade = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)
    pitch = Column(Float, nullable=True)
    roll = Column(Float, nullable=True)

class ObstaculoDetectado(Base):
    __tablename__ = "obstaculos_detectados"
    id = Column(Integer, primary_key=True, index=True)
    missao_id = Column(Integer, index=True)
    track_id = Column(Integer, index=True) # Para atualizar o mesmo objeto se ele permanecer na tela
    classe = Column(String, nullable=False)
    risco_max = Column(String, nullable=False)
    confianca_max = Column(Float, nullable=True)
    proximidade_min = Column(Float, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    primeira_deteccao = Column(Float, nullable=False)
    ultima_deteccao = Column(Float, nullable=False)

class Configuracao(Base):
    __tablename__ = "configuracoes"
    id = Column(Integer, primary_key=True, index=True)
    # Salvamos os fields de config como um JSON inteiro para facilitar compatibilidade com Pydantic
    settings_json = Column(JSON, nullable=False)

class AvaliacaoModelo(Base):
    __tablename__ = "avaliacoes_modelo"
    id = Column(Integer, primary_key=True, index=True)
    data_avaliacao = Column(String, nullable=False) # ISO format
    modelo = Column(String, nullable=True)
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    precision_global = Column(Float, nullable=True)
    recall_global = Column(Float, nullable=True)
    total_fp = Column(Integer, nullable=True)
    total_fn = Column(Integer, nullable=True)
    metricas_classe = Column(JSON, nullable=True)

# Helper para obter a sessão do BD
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
