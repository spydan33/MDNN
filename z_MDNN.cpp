#include "vector_space.cpp"
#include "MNISTImageReader.cpp"
#include "MNISTLabelReader.cpp"
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <exception>
using namespace std;

class MDNN
{
    private:
        bool stuff;
        vector_space vs;
        int inputs = 732;
        int outputs_num = 10;
        vector<float> output;
        vector<node*> output_nodes; //for testing and not required
        vector<node*> nodes_fired; //for learning and not required
        float action_potential_threshold = 3;
        bool input_layer = true;
        int neurons_used = 0;
        float learning_rate = 0.001;
        int back_prop_iteration = 0;


    public:
        bool verbose = false;
        MDNN()
        {
            int n_vect;
            bool vs_loaded = vs.loaded();
            if(!vs_loaded)
            {/*
                cout << "Enter Cell size: \n";
                
                cin >> cell_size;
                vs.set_cell_size(cell_size);
                */
                cout << "Please enter the number of neurons to begin: \n";
                cin >> n_vect;
                vs.generate_vectors(n_vect);
                vs.save();
            }
        };
        
        vector<float>* get_output() 
        {
            return &output;
        }
        vector<float> cascade(vector<float> input)
        {
            vector<node*> fireing_nodes;
            string ent;
            int mod;
            float i = 0;
            float weight;
            node fireing_neuron;
            for(float &i_input : input)
            {
                if(input_layer)
                {
                    if(static_cast<int>(i) % 2 == 0)
                    {
                        mod = -1;
                    }
                    else
                    {
                        mod = 1;
                    }
                    fireing_neuron = {(normalize((i * mod)) * 10),(normalize((i * mod)) * 10),(normalize((i * mod)) * 10),1};
                    fireing_neuron.fired = true;
                    auto neurons = vs.radius_search(fireing_neuron);
                    for(node* neuron : neurons)
                    {
                        float distance = vs.get_distance(fireing_neuron,neuron);
                        if(distance == 0)
                        {
                            continue;
                        }
                        weight = 1 / (distance);
                        neuron->used = true;
                        neuron->value = (((weight * neuron->s) * (i_input) /** 100*/) + neuron->bias);
                        if(!neuron->fired)
                        {
                            if(!neuron->staged)
                            {
                                neuron->staged = true;
                                fireing_nodes.push_back(neuron);
                            }
                        }
                    }
                }
                i ++;
            }
            
            //cout << "Next layer size: ";
            //cout << fireing_nodes.size();
            //cout << "\n";
            return cascade(fireing_nodes);
        }
    
        vector<float> cascade(vector<node*> &input)
        {
            //cout << "next layer hit \n";
            vector<node*> fireing_nodes;
            fireing_nodes.clear();
            string ent;
            int mod;
            float weight;
            for(node *fireing_node : input)
            {
                bool result1 = (fireing_node->bias < 0) ? (fireing_node->value < fireing_node->bias) : (fireing_node->value > fireing_node->bias);
                if(!result1 /*&& (fireing_node->value > action_potential_threshold || fireing_node->value < -action_potential_threshold)*/ )//newly added can be removed if not having the same input location
                {
                    fireing_node->fired = true;
                    nodes_fired.push_back(fireing_node);
                    neurons_used += 1;
                    vector<node *> next_layer = vs.radius_search(fireing_node);
                    for(node* next_node : next_layer)
                    {
                        if(!next_node->fired)
                        {
                            if(relu(fireing_node->value) == 0)
                            {
                                continue;
                            }
                            float distance = vs.get_distance(fireing_node,next_node);
                            if(distance == 0)
                            {
                                continue;
                            }
                            weight = 1 / (distance);
                            fireing_node->used = true;
                            next_node->value += normalize((((weight * next_node->s) * fireing_node->value))) + fireing_node->bias;
                            if(!next_node->staged)
                            {
                                next_node->staged = true;
                                bool result2 = (next_node->bias < 0) ? (next_node->value < next_node->bias) : (next_node->value > next_node->bias);
                                if(!result2)
                                {
                                    fireing_nodes.push_back(next_node);//auto *output_ptr = &output;
                                    if(output.size() == outputs_num)
                                    {
                                        output.erase(output.begin());
                                        output.push_back(sigmoid(next_node->value));
                                        output_nodes.erase(output_nodes.begin());
                                        output_nodes.push_back(next_node);
                                    }
                                    else
                                    {
                                        output.push_back(sigmoid(next_node->value));
                                        output_nodes.push_back(next_node);
                                    }
                                }
                            }
                        }
                    }
                } 
            }
            
            //cout << "Next layer size: ";
            //cout << fireing_nodes.size();
            //cout << "\n";
            if(fireing_nodes.size() > 0)
            {
                return cascade(fireing_nodes);
            }
            else
            {
                if(verbose || back_prop_iteration % 100 == 0)
                {
                    print_neurons_used();
                    cout << "---output neurons--- \n";
                    for(node* output_node : output_nodes)
                    {
                        cout << "[" << output_node->x << "," << output_node->y << "," << output_node->z << "] value: "<< output_node->value <<" \n";
                    }
                    cout << "------------------- \n";
                }
                return output;
            }
        }

        // Loss calculation
        float calc_loss(vector<float> expected_output)
        {
            // Mean square loss
            float sum = 0;
            for (int i = 0; i < expected_output.size(); i++)
            {
                float calc_val = (output[i] - expected_output[i]);
                sum += (calc_val * calc_val);
            }
            if(verbose)
            {
                cout << "loss: " << (sum / expected_output.size()) << "\n";
            }
            return (sum / expected_output.size());
        }
        void back_propagation(std::vector<float> expected_output)
        {
            // Basic mean squared error gradient descent on the output nodes.
            if (output_nodes.size() != expected_output.size())
            {
                return; // Mismatched data
            }

            float loss = calc_loss(expected_output);

            for (size_t i = 0; i < output_nodes.size(); ++i)
            {
                node* out = output_nodes[i];
                float error = output[i] - expected_output[i];

                // Update bias directly using the error term
                out->bias -= learning_rate * error;

                // Adjust position based on neighbouring nodes
                auto neighbours = vs.radius_search_all(out);
                for (node* prev : neighbours)
                {
                    float distance = vs.get_distance(out, prev);
                    if (distance == 0.0f) continue;

                    float weight_sign = out->s;
                    float grad = error * prev->value * weight_sign / std::pow(distance, 3);

                    out->x -= learning_rate * grad * (out->x - prev->x);
                    out->y -= learning_rate * grad * (out->y - prev->y);
                    out->z -= learning_rate * grad * (out->z - prev->z);
                }

                out->fired = false;
                out->staged = false;
            }

            nodes_fired.clear();
            back_prop_iteration += 1;

            if(back_prop_iteration % 40 == 0)
            {
                std::cout << "loss: " << loss << "\n";
                vs.save();
            }

            vs.re_map();

            if(back_prop_iteration % 100 == 0)
            {
                std::cout << "loss: " << loss << "\n";
                vs.prune();
            }
        }
        float not_zero(float value) 
        {
            return (value == 0) ? 1 : value;
        }
        float sigmoid(float value) 
        {
            // Sigmoid function to normalize the value to the range [0, 1]
            return 1.0 / (1.0 + std::exp(-value));
        }
        float relu(float x)
        {
            return (x > 0) ? x : 0;
        }
        float normalize(float value) 
        {
            // Normalizes the value to the range [-1, 1]
            return std::tanh(value);
        }

        void test()
        {
            cout << "complete \n";
        }

        void clearNeurons()
        {
            neurons_used = 0;
        }
        void prune()
        {
            vs.prune();
        }

        void print_neurons_used()
        {
            cout << "neurons used "<< neurons_used << "\n";
        }
        std::vector<float> generateRandomFloats(int n)
        {
            std::vector<float> vec(n);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            
            for(int i = 0; i < n; ++i)
            {
                vec[i] = dis(gen);
            }
            
            return vec;
        }
        std::vector<float> generateOneHotVector(int n)
        {
            std::vector<float> vec(n, 0.0f);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(0, n - 1);

            int randomIndex = dis(gen);
            vec[randomIndex] = 1.0f;
            
            return vec;
        }
        int indexOfMaxValue(const std::vector<float>& vec)
        {
            if (vec.empty())
            {
                throw std::out_of_range("The vector is empty.");
            }
            
            int maxIndex = 0;
            for (int i = 1; i < vec.size(); ++i)
            {
                if (vec[i] > vec[maxIndex])
                {
                    maxIndex = i;
                }
            }
            return maxIndex;
        }


};

int main() 
{
    MNISTImageReader reader("train-images.idx3-ubyte");
    const auto &images = reader.getImages();
    //float x = 0.1,y = 0.5,z = 0.6;
    MNISTLabelReader reader_2("train-labels.idx1-ubyte");
    auto one_hot_labels = reader_2.getOneHotLabels();

    MDNN nn;
    nn.test();
    //cout << "Enter start value: \n";
    //cin >> x >> y >> z;
    
    ///auto in_vec = nn.generateRandomFloats(732);
    //auto out_got_vec = nn.generateRandomFloats(732);
    /*
    for(int i = 0; i < 1; i ++)
    {*/
        
        auto start1 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < images.size(); i ++)
        {
            try
            {
                if(i > 70000)
                {
                    break;
                }
                //auto start = std::chrono::high_resolution_clock::now();
                //auto ret = nn.cascade({x, y, z});
                auto ret = nn.cascade(images[i].pixels);
                //auto stop = std::chrono::high_resolution_clock::now();
                if(nn.verbose)
                {
                    cout << "output:[";
                    for(float out : ret)
                    {
                        cout << out << ",";
                    }
                    cout << "]\n";  
                }
                else
                {
                    if(i % 100 == 0)
                    {
                        nn.print_neurons_used();
                        cout << "output:[";
                        for(float out : ret)
                        {
                            cout << out << ",";
                        }
                        cout << "]\n"; 
                        
                        int guess = (nn.indexOfMaxValue(ret)); 
                        
                        cout << "guess:["<< guess <<"]\n"; 
                    }
                }
                if(nn.verbose)
                {
                    cout << "expected: [";
                    for(float out_1 : one_hot_labels[i])
                    {
                        cout << out_1 << ",";
                    }
                    cout << "]\n";
                }
                else
                {
                    if(i % 100 == 0)
                    {
                        cout << "expected: [";
                        for(float out_1 : one_hot_labels[i])
                        {
                            cout << out_1 << ",";
                        }
                        cout << "]\n";
                    }
                }
                //nn.print_neurons_used();
                nn.clearNeurons();
                //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                //std::cout << "Time taken to forward: " << duration.count() << " milliseconds" << std::endl;
                nn.back_propagation(one_hot_labels[i]);
                /*
                cout << "expected: [";
                for(float out_1 : one_hot_labels[i])
                {
                    cout << out_1 << ",";
                }
                cout << "]\n";*/
                if(i % 5000 == 0)
                {
                    //nn.prune();
                }
            }
            catch (const std::exception &e)
            {
                cout << "An error occurred: interation:" << i << " :" << e.what() << std::endl;
                break;
            }

        }
        auto stop1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
        std::cout << "Time taken to backward: " << duration1.count() << " milliseconds" << std::endl;
    //}
    cout << "Press anything to end: \n";
    string finished;
    cin >> finished;
    return 0;
}