
import sqlite3
from annoy import AnnoyIndex
connection = sqlite3.connect('biases.db')
cursor = connection.cursor()
cursor.execute("select * from bias where not activated")
ret = cursor.fetchall()
prune = False
for row in ret:
    print(row)


index = AnnoyIndex(3, 'euclidean')
for i in range(100):
    print(index.get_item_vector(i))
