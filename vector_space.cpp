#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include "hashKey.cpp"
#include <chrono>
#include <fstream>

using namespace std;

class vector_space 
{
    private:
        unordered_map<tuple<int, int, int, int>, vector<node*>, KeyHash, KeyEqual> vect_map;
        int vector_space_range_low; //clean up: false 
        int vector_space_range_high; //clearn up: false
        string filename = "vectorSpace.vs";
        std::mt19937 gen;
        bool loaded_bol;
        int loaded_neurons = 0;

        
        float cluster_cell_size = 2; //5
        float activation_distance = 1.9; //4
        bool verbose = false;

        
        tuple<int, int, int, int> hashKey(const node& vec) //remarkable round down function for hashing query vectors and saved vectors
        {
            if(verbose)
            {
                //cout << "Clusert size at hash key: " << cluster_cell_size << "\n";
            }
            return {static_cast<int>(floor(vec.x / cluster_cell_size)),
                    static_cast<int>(floor(vec.y / cluster_cell_size)),
                    static_cast<int>(floor(vec.z / cluster_cell_size)),
                    static_cast<int>(floor(vec.s / cluster_cell_size))};
        }
        tuple<int, int, int, int> hashKey(const node* vec) //remarkable round down function for hashing query vectors and saved vectors
        {
            if(verbose)
            {
                //cout << "Clusert size at hash key: " << cluster_cell_size << "\n";
            }
            return {static_cast<int>(floor(vec->x / cluster_cell_size)),
                    static_cast<int>(floor(vec->y / cluster_cell_size)),
                    static_cast<int>(floor(vec->z / cluster_cell_size)),
                    static_cast<int>(floor(vec->s / cluster_cell_size))};
        }

        bool fileExists() 
        {
            std::ifstream file(filename);
            return file.good();
        }

        float get_random_number(float mean, float stddev) 
        {
            // Normal distribution generator
            std::normal_distribution<> distr(mean, stddev); // Define the mean and standard deviation
            return distr(gen); // Generate and return the random number
        }
        node get_random_vector()
        {
            
            // Function to get a random vector based on normal distribution
            float mean = 0.0f; // Center the distribution around the origin
            float stddev = 2.5f; // Define the spread of the distribution
            return {
                get_random_number(mean, stddev), 
                get_random_number(mean, stddev), 
                get_random_number(mean, stddev), 
                get_suppression()
            };
        }
        /*
        float get_random_number(int low, int high) 
        {
            // Random number generator
            std::uniform_real_distribution<> distr(low, high); // Define the range
            return distr(gen); // Generate and return the random number
        }


        node get_random_vector()
        {
            return {get_random_number(vector_space_range_low,vector_space_range_high),get_random_number(vector_space_range_low,vector_space_range_high),get_random_number(vector_space_range_low,vector_space_range_high),get_suppression()};
        }*/
        float get_suppression() 
        {
            // Random number generator
            std::uniform_int_distribution<> distr(0, 1); // Define the range to generate either 0 or 1
            int random_bit = distr(gen); // Generate either 0 or 1
            return random_bit == 0 ? -1 : 1; // Map 0 to -1, and 1 to 1
        }

        

        bool check_distance(const node& n1, const node& n2)
        {
            float dx = n1.x - n2.x;
            float dy = n1.y - n2.y;
            float dz = n1.z - n2.z;
            auto distance = get_distance(n1,n2);
            if(activation_distance > distance)
            {
                //cout << "In range: " << distance << "\n";
                return true;
            }
            else
            {
                //cout << "Failed range: " << distance << "\n";
                return false;
            }
        }

        bool check_distance(const node* n1, const node* n2)
        {
            float dx = n1->x - n2->x;
            float dy = n1->y - n2->y;
            float dz = n1->z - n2->z;
            auto distance = get_distance(n1,n2);
            if(activation_distance > distance)
            {
                //cout << "In range: " << distance << "\n";
                return true;
            }
            else
            {
                //cout << "Failed range: " << distance << "\n";
                return false;
            }
        }
        
        bool check_distance(const node& n1, const node* n2)
        {
            float dx = n1.x - n2->x;
            float dy = n1.y - n2->y;
            float dz = n1.z - n2->z;
            auto distance = get_distance(n1,n2);
            if(activation_distance > distance)
            {
                //cout << "In range: " << distance << "\n";
                return true;
            }
            else
            {
                //cout << "Failed range: " << distance << "\n";
                return false;
            }
        }
        
        bool check_distance(const node* n1, const node& n2)
        {
            float dx = n1->x - n2.x;
            float dy = n1->y - n2.y;
            float dz = n1->z - n2.z;
            auto distance = get_distance(n1,n2);
            if(activation_distance > distance)
            {
                //cout << "In range: " << distance << "\n";
                return true;
            }
            else
            {
                //cout << "Failed range: " << distance << "\n";
                return false;
            }
        }

    public:
        vector_space(float cs) 
            : cluster_cell_size(cs), 
              vector_space_range_low(-100), 
              vector_space_range_high(100),
              loaded_bol(false),
              gen(std::random_device{}()) {check_load();}

        vector_space(float cs, int low, int high) 
            : cluster_cell_size(cs), 
              vector_space_range_low(low), 
              vector_space_range_high(high),
              loaded_bol(false),
              gen(std::random_device{}()) {check_load();}

        vector_space() :
            vector_space_range_low(-100), 
            vector_space_range_high(100),
            loaded_bol(false) {check_load();}


        
        float get_distance(const node& n1, const node& n2)
        {
            float dx = n1.x - n2.x;
            float dy = n1.y - n2.y;
            float dz = n1.z - n2.z;
            float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            return distance;
        }

        float get_distance(const node* n1, const node* n2)
        {
            float dx = n1->x - n2->x;
            float dy = n1->y - n2->y;
            float dz = n1->z - n2->z;
            float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            return distance;
        }

        float get_distance(const node* n1, const node& n2)
        {
            float dx = n1->x - n2.x;
            float dy = n1->y - n2.y;
            float dz = n1->z - n2.z;
            float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            return distance;
        }

        float get_distance(const node& n1, const node* n2)
        {
            float dx = n1.x - n2->x;
            float dy = n1.y - n2->y;
            float dz = n1.z - n2->z;
            float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            return distance;
        }

        void set_cell_size(float cs)
        {
            cluster_cell_size = cs;
        }
        void add_vector(node& vec) 
        {
            auto key = hashKey(vec);
            if(verbose)
            {
                //cout << "add defined vector: " << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.s << "\n";
            }
            node* new_node = new node(vec);
            new_node->bias = vec.bias;
            new_node->used = vec.used;
            vect_map[key].push_back(new_node);
        }

        void add_vector() 
        {
            node vec = get_random_vector();
            vec.bias = get_random_number(0,5);
            if(verbose)
            {
                //cout << "add random vector: " << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.s << "\n";
            }
            auto key = hashKey(vec);
            node* new_node = new node(vec);
            vect_map[key].push_back(new_node);
        }

        void generate_vectors(int n_vect)
        {
            for(int i = 0; i < n_vect; i ++)
            {
                add_vector();
            }
        }
        vector<node*> radius_search_all(node* query_node)// contains refference to the dimension, d = 4
        {
            //cout << "pointer search";
            vector<node*> total_nodes;
            auto key = hashKey(query_node);
            auto &cluster = vect_map[key];
            if (cluster.empty()) 
            {
                if(verbose)
                {
                    cout << "throw error? no neurons in search cluster \n";
                }
                //cout << "throw error? no neurons in search cluster \n";
            }
            for (node* neuron : cluster) 
            {
                if(check_distance(query_node,neuron))
                {
                    total_nodes.push_back(neuron);
                }
            }
            for (float x = -cluster_cell_size; x <= cluster_cell_size; x += cluster_cell_size) 
            {
                for (float y = -cluster_cell_size; y <= cluster_cell_size; y += cluster_cell_size) 
                {
                    for (float z = -cluster_cell_size; z <= cluster_cell_size; z += cluster_cell_size) 
                    {
                        auto key1 = hashKey(node {query_node->x + x,query_node->y + y,query_node->z + z,1});
                        auto& near_cluster_1 = vect_map[key1];
                        for (node* neuron : near_cluster_1) 
                        {
                            if(check_distance(query_node,neuron))
                            {
                                total_nodes.push_back(neuron);
                            }
                        }
                        auto key2 = hashKey(node {query_node->x + x,query_node->y + y,query_node->z + z,-1});
                        auto& near_cluster_2 = vect_map[key2];
                        for (node* neuron : near_cluster_2) 
                        {
                            if(check_distance(query_node,neuron))
                            {
                                total_nodes.push_back(neuron);
                            }
                        }
                    }
                }
            }
            return total_nodes;
        }
        vector<node*> radius_search(node query_node)// contains refference to the dimension, d = 4
        {
            //cout << "hard radius search \n";
            vector<node*> total_nodes;
            auto key = hashKey(query_node);
            
            auto &cluster = vect_map[key];
            if (cluster.empty()) 
            {
                if(verbose)
                {
                    cout << "throw error? no neurons in search cluster \n";
                }
                //cout << "throw error? no neurons in search cluster \n";
            }
            for (node* neuron : cluster) 
            {
                if(!neuron->fired && check_distance(query_node,neuron))
                {
                    total_nodes.push_back(neuron);
                }
                else
                {
                    if(!neuron->fired)
                    {
                        //cout << "blocked fired node! search 1\n";
                    }
                }
            }
            for (float x = -cluster_cell_size; x <= cluster_cell_size; x += cluster_cell_size) 
            {
                for (float y = -cluster_cell_size; y <= cluster_cell_size; y += cluster_cell_size) 
                {
                    for (float z = -cluster_cell_size; z <= cluster_cell_size; z += cluster_cell_size) 
                    {
                        auto key1 = hashKey(node {query_node.x + x,query_node.y + y,query_node.z + z,1});
                        auto& near_cluster_1 = vect_map[key1];
                        for (node* neuron : near_cluster_1) 
                        {
                            if(!neuron->fired && check_distance(query_node,neuron))
                            {
                                total_nodes.push_back(neuron);
                            }
                        }
                        auto key2 = hashKey(node {query_node.x + x,query_node.y + y,query_node.z + z,-1});
                        auto& near_cluster_2 = vect_map[key2];
                        for (node* neuron : near_cluster_2) 
                        {
                            if(!neuron->fired && check_distance(query_node,neuron))
                            {
                                total_nodes.push_back(neuron);
                            }
                        }
                    }
                }
            }
            if(verbose)
            {
                for (node* neuron : total_nodes) 
                {
                    std::cout << "node: [" << neuron->x<< ", " << neuron->y << ", " << neuron->z << ", " << neuron->s << "] \n";
                    //std::cout << "node: [" << neuron.x<< ", " << neuron.y << ", " << neuron.z << ", " << neuron.s << "] \n";
                }
            }
            return total_nodes;
        }

        vector<node*> radius_search(node* query_node)// contains refference to the dimension, d = 4
        {
            //cout << "pointer search";
            vector<node*> total_nodes;
            auto key = hashKey(query_node);
            
            auto &cluster = vect_map[key];
            if (cluster.empty()) 
            {
                if(verbose)
                {
                    cout << "throw error? no neurons in search cluster \n";
                }
                //cout << "throw error? no neurons in search cluster \n";
            }
            for (node* neuron : cluster) 
            {
                if(!neuron->fired && check_distance(query_node,neuron))
                {
                    total_nodes.push_back(neuron);
                }
            }
            for (float x = -cluster_cell_size; x <= cluster_cell_size; x += cluster_cell_size) 
            {
                for (float y = -cluster_cell_size; y <= cluster_cell_size; y += cluster_cell_size) 
                {
                    for (float z = -cluster_cell_size; z <= cluster_cell_size; z += cluster_cell_size) 
                    {
                        auto key1 = hashKey(node {query_node->x + x,query_node->y + y,query_node->z + z,1});
                        auto& near_cluster_1 = vect_map[key1];
                        for (node* neuron : near_cluster_1) 
                        {
                            if(!neuron->fired &&check_distance(query_node,neuron))
                            {
                                total_nodes.push_back(neuron);
                            }
                        }
                        auto key2 = hashKey(node {query_node->x + x,query_node->y + y,query_node->z + z,-1});
                        auto& near_cluster_2 = vect_map[key2];
                        for (node* neuron : near_cluster_2) 
                        {
                            if(!neuron->fired && check_distance(query_node,neuron))
                            {
                                total_nodes.push_back(neuron);
                            }
                        }
                    }
                }
            }
            return total_nodes;
        }
        
        vector<node*> fired_radius_search(node* query_node)// contains refference to the dimension, d = 4
        {
            //cout << "pointer search";
            vector<node*> total_nodes;
            auto key = hashKey(query_node);
            
            auto &cluster = vect_map[key];
            if (cluster.empty()) 
            {
                if(verbose)
                {
                    cout << "throw error? no neurons in search cluster \n";
                }
                //cout << "throw error? no neurons in search cluster \n";
            }
            for (float x = -cluster_cell_size; x <= cluster_cell_size; x += cluster_cell_size) 
            {
                for (float y = -cluster_cell_size; y <= cluster_cell_size; y += cluster_cell_size) 
                {
                    for (float z = -cluster_cell_size; z <= cluster_cell_size; z += cluster_cell_size) 
                    {
                        auto key1 = hashKey(node {query_node->x + x,query_node->y + y,query_node->z + z,1});
                        auto& near_cluster_1 = vect_map[key1];
                        for (node* neuron : near_cluster_1) 
                        {
                            if(neuron->fired && check_distance(query_node,neuron))
                            {
                                total_nodes.push_back(neuron);
                            }
                        }
                        auto key2 = hashKey(node {query_node->x + x,query_node->y + y,query_node->z + z,-1});
                        auto& near_cluster_2 = vect_map[key2];
                        for (node* neuron : near_cluster_2) 
                        {
                            if(neuron->fired && check_distance(query_node,neuron))
                            {
                                total_nodes.push_back(neuron);
                            }
                        }
                    }
                }
            }
            for (node* neuron : cluster) 
            {
                if(neuron->fired && check_distance(query_node,neuron))
                {
                    total_nodes.push_back(neuron);
                }
            }
            return total_nodes;
        }

        void load(const std::string& filename)
        {
            std::cout << "---load func hit--- \n";
            std::ifstream file(filename);
            if (!file.is_open()) 
            {
                std::cerr << "Failed to open file for reading." << std::endl;
                return;
            }

            vect_map.clear();
            int loaded_neurons = 0;
            int x, y, z, s;
            float node_x, node_y, node_z, node_s, bias;
            bool fired,used;

            while (file >> x >> y >> z >> s >> node_x >> node_y >> node_z >> node_s >> fired >> bias >> used) 
            {
                node new_node;
                new_node.x = node_x;
                new_node.y = node_y;
                new_node.z = node_z;
                new_node.s = node_s;
                new_node.fired = fired;
                new_node.bias = bias;
                new_node.used = used;
                loaded_neurons += 1;
                add_vector(new_node);
            }

            std::cout << "Loaded: " << loaded_neurons << " neurons \n";
            file.close();
        }
        void save() 
        {
            std::ofstream file(filename);
            if (!file.is_open()) 
            {
                std::cerr << "Failed to open file for writing." << std::endl;
                return;
            }
            for (const auto& pair : vect_map) 
            {
                for (const node* node : pair.second) 
                {
                    file << std::get<0>(pair.first) << " " << std::get<1>(pair.first) << " "
                        << std::get<2>(pair.first) << " " << std::get<3>(pair.first) << " "
                        << node->x << " " << node->y << " " << node->z << " " << node->s << " "
                        << node->fired << " " << node->bias << " " << node->used <<"\n";
                }
            }
            
            //cout << "saved file \n";
            file.close();
        }
        void re_map() 
        {
            auto old_map = vect_map;
            vect_map.clear();
            for (auto& pair : old_map) 
            {
                for (node * node_old : pair.second) 
                {
                    node new_node = {node_old->x,node_old->y,node_old->z,node_old->s};
                    new_node.bias = node_old->bias;
                    new_node.used = node_old->used;
                    add_vector(new_node);
                }
            }
        }
        void prune() 
        {
            std::ofstream file(filename);
            if (!file.is_open()) 
            {
                std::cerr << "Failed to open file for writing." << std::endl;
                return;
            }
            int not_used_count = 0;
            for (const auto& pair : vect_map) 
            {
                for (const node* node : pair.second) 
                {
                    if(node->used)
                    {
                        file << std::get<0>(pair.first) << " " << std::get<1>(pair.first) << " "
                        << std::get<2>(pair.first) << " " << std::get<3>(pair.first) << " "
                        << node->x << " " << node->y << " " << node->z << " " << node->s << " "
                        << node->fired << " " << node->bias << " " << node->used <<"\n";
                    }
                    else
                    {
                        not_used_count += 1;
                    }
                }
            }
            
            cout << "Not Used Count: "<< not_used_count <<"\n";
            file.close();
            load(filename); //comment out to return to previous
        }


        vector<node *> get_all()
        {
            vector<node *> all;
            for (const auto& pair : vect_map) 
            {
                for (node* node : pair.second) 
                {
                    all.push_back(node);
                }
            }
            return all;
        }
        void check_load()
        {
            if (fileExists()) 
            {
                float cell_size;
                load(filename);
                cout << "Loaded file \n";
                loaded_bol = true;
            }
            return;
        }

        bool loaded()
        {
            return loaded_bol;
        }
};

/*
int main() 
{
    int n_vect;
    float cell_size;
    float x,y,z,s;
    vector_space vs;
    auto vs_loaded = vs.loaded();
    if(!vs_loaded)
    {
        cout << "Enter Cell size: \n";
        
        cin >> cell_size;
        vs.set_cell_size(cell_size);
        cout << "Please enter the number of neurons to begin: \n";
        cin >> n_vect;
        vs.generate_vectors(n_vect);
        vs.save();
    }
    while(true)
    {
        //cout << "Vector Count: " << n_vect << "\n";
        cout << "Enter query vector: \n";
        cin >> x >> y >> z >> s ;
        node query = {x, y, z, s};
        auto start = std::chrono::high_resolution_clock::now();
        vector<node> nearest = vs.radius_search(query);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken to search: " << duration.count() << " milliseconds" << std::endl;
    }
	return 0;
}
*/
