import sqlite3
from pathlib import Path
from datetime import datetime


# ============================================================
# DATABASE PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"

DATABASE_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "analysis_history.db"


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            title TEXT,
            url TEXT,
            prediction TEXT,
            trust_score REAL,
            trust_level TEXT,
            risk_level TEXT
        )
    """)

    connection.commit()
    connection.close()


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_analysis(result, title="", url=""):

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO analysis_history (
            timestamp,
            title,
            url,
            prediction,
            trust_score,
            trust_level,
            risk_level
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        title,
        url,
        result["prediction"],
        result["trust_score"],
        result["trust_level"],
        result["risk_level"]
    ))

    connection.commit()
    connection.close()


# ============================================================
# GET ANALYSIS HISTORY
# ============================================================

def get_analysis_history():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            title,
            url,
            prediction,
            trust_score,
            trust_level,
            risk_level
        FROM analysis_history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows


# ============================================================
# INITIALIZE DATABASE WHEN MODULE LOADS
# ============================================================

initialize_database()