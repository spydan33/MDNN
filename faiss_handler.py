import faiss
import numpy as np
import traceback
class faiss_handler():
    def __init__(self,index = False):
        self.index = index
        self.return_np = False
        self.path_to_index = False
        self.path_to_np = False
        self.pre_index = False
        self.vector_list = []
    def set_index_path(self, input_path_to_index):
        self.path_to_index = input_path_to_index
    def set_list_path(self, path_to_list):
        self.path_to_np = path_to_list
    def get_vector(self,vector_id):
        try:
            ret_vector = self.vector_list[vector_id]
            return ret_vector
        except Exception as e:
            traceback.print_exc()
            print("An error occurred:", e)
            return False
    
    def radius_search_id(self,vector_id,radius,exlude_ids = False):
        try:
            if(self.index == False):
                raise Exception("Index not Initalized")
            query_vector = self.vector_list[vector_id]
            if(not exlude_ids == False):
                sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(exlude_ids))
                params=faiss.SearchParametersIVF(
                    nprobe=100,
                    sel=sel,
                    max_codes= self.index.nlist
                )
                ret_vectors = self.index.range_search(np.array([query_vector]), radius, params=params)
            else:
                params=faiss.SearchParametersIVF(
                    nprobe=100,
                    max_codes= self.index.nlist
                )
                ret_vectors = self.index.range_search(np.array([query_vector]), radius, params=params)
            return self.__return_array(np.array([ret_vectors[2].astype(int),ret_vectors[1]]))
            # Filter based on distance
        except Exception as e:
            traceback.print_exc()
            print("An error occurred:", e)
            return False
    
    def radius_search_vector(self,query_vector,radius,exlude_ids = False):
        try:
            if(self.index == False):
                raise Exception("Index not Initalized")
            if(not exlude_ids == False):
                sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(exlude_ids))
                params=faiss.SearchParametersIVF(
                    nprobe=100,
                    sel=sel,
                    max_codes= self.index.nlist
                )
                ret_vectors = self.index.range_search(np.array([query_vector]), radius, params=params)
            else:
                params=faiss.SearchParametersIVF(
                    nprobe=100,
                    max_codes= self.index.nlist
                )
                ret_vectors = self.index.range_search(np.array([query_vector]), radius, params=params)
            return self.__return_array(np.array([ret_vectors[2].astype(int),ret_vectors[1]]))
            # Filter based on distance
        except Exception as e:
            traceback.print_exc()
            print("An error occurred:", e)
            return False
    def get_distance(self,vector_id_1,vector_id_2):
        try:
            
            if  not isinstance(vector_id_1, list):
                vector_1 = np.array(self.get_vector(vector_id_1))
            else:
                vector_1 = vector_id_1
            if  not isinstance(vector_id_2, list):
                vector_2 = np.array(self.get_vector(vector_id_2))
            else:
                vector_2 = venctor_id_2
            return np.linalg.norm(vector_1 - vector_2)
        except Exception as e:
            traceback.print_exc()
            print("An error occurred:", e)
            return False

    def set_pre_index(self,dim,nlist):
        try:
            if(self.path_to_index == False):
                raise Exception("path not set")
            # Create an index for the quantizer
            quantizer = faiss.IndexFlatL2(dim)
            # Create the IVFFlat index
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            # Filter based on distance
            self.pre_index = index
        except Exception as e:
            traceback.print_exc()
            print("An error occurred:", e)
            return False
    def load(self):
        try:
            if(self.path_to_index == False):
                raise Exception("index path not set")
            if(self.path_to_np == False):
                raise Exception("index path not set")
            self.index = faiss.read_index(self.path_to_index)
            self.vector_list = np.load(self.path_to_np)
        except Exception as e:
            traceback.print_exc()
            print("An error occurred:", e)
            return False
    def save(self):
        try:
            if(self.path_to_index == False):
                raise Exception("path not set")
            if(self.pre_index == False):
                raise Exception("pre index not set")
            
            faiss.write_index(self.pre_index, self.path_to_index)
            np.save(self.path_to_np,self.vector_list)
        except Exception as e:
            traceback.print_exc()
            print("An error occurred:", e)
            return False
    def add(self,vectors):
        try:
            if(self.pre_index == False):
                raise Exception("pre index not set")
            if(self.index != False):
                raise Exception("Index is already built")
            if isinstance(vectors, list):
                self.pre_index.train(np.array(vectors))
                self.pre_index.add(np.array(vectors))
                self.vector_list = np.array(vectors)
            else:
                raise Exception("Vectors must be list")
        except Exception as e:
            traceback.print_exc()
            print("An error occurred:", e)
            return False
    def __return_array(self,array_to_ret):
        if(self.return_np):
            return array_to_ret
        else:
            return array_to_ret.tolist()
