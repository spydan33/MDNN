#include <iostream>
#include <vector>
#include <exception>
#include <faiss>

class FaissHandler 
{
    public:
        FaissHandler(bool index = false) : 
            index(index), return_np(false), path_to_index(""), path_to_np(""), pre_index(false) {}

        void setIndexPath(const std::string& inputPathToIndex) 
        {
            path_to_index = inputPathToIndex;
        }

        void setListPath(const std::string& pathToList) 
        {
            path_to_np = pathToList;
        }

        std::vector<float> getVector(int vectorId) 
        {
            try 
            {
                // Assuming vector_list is a 2D vector for simplicity: std::vector<std::vector<float>>
                return vector_list.at(vectorId);
            } 
            catch (const std::exception& e) 
            {
                std::cerr << "An error occurred: " << e.what() << std::endl;
                return std::vector<float>(); // Return an empty vector on failure
            }
        }

    private:
        bool index;
        bool return_np;
        std::string path_to_index;
        std::string path_to_np;
        bool pre_index;
        std::vector<std::vector<float>> vector_list; // Example of storing vectors
};

