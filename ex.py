from faiss_handler import faiss_handler
import numpy as np

fs = faiss_handler()
fs.set_index_path("neurons1.index")
fs.set_list_path("neurons_list1.npy")
fs.load()


ret = fs.radius_search_id(1,6)
print(ret)

dist = fs.get_distance(1,3835)
print(dist)
print(dist ** 2)