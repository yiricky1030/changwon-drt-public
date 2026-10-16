"""Database connection factory."""
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL


def get_engine() -> Engine:
    """Create a SQLAlchemy engine from environment variables."""
    url = URL.create("postgresql+psycopg2", username=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"], host=os.getenv("POSTGRES_HOST", "localhost"), port=int(os.getenv("POSTGRES_PORT", "5432")), database=os.environ["POSTGRES_DB"])
    return create_engine(url)

