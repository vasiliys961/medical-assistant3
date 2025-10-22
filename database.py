import sqlite3

def init_database():
    conn = sqlite3.connect('medical_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            raw_text TEXT,
            structured_note TEXT,
            gdoc_url TEXT,
            diagnosis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
    ''')
    conn.commit()
    conn.close()

def save_medical_note(patient_id, raw_text, structured_note, gdoc_url, diagnosis):
    conn = sqlite3.connect('medical_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO patient_notes (patient_id, raw_text, structured_note, gdoc_url, diagnosis)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient_id, raw_text, structured_note, gdoc_url, diagnosis))
    conn.commit()
    conn.close()