import sqlite3
class load:
    def create_table():
        connection = sqlite3.connect('biases.db')
        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS bias (bias_id INTEGER PRIMARY KEY, bias INTEGER, vector_id INTEGER, suppress boolean, input boolean default false,activated boolean default false)')
        cursor.execute('CREATE INDEX IF NOT EXISTS vector_id_index ON bias(vector_id)')
        connection.commit()
        connection.close()
    def reset_neurons():
        connection = sqlite3.connect('biases.db')
        cursor = connection.cursor()
        cursor.execute('UPDATE bias SET activated = false')
        connection.commit()
        connection.close()