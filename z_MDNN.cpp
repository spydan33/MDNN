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
        void back_propagation(vector<float> expected_output)
        {
            float loss = calc_loss(expected_output); // Calculate loss once before the loop
            float decay_factor = 0.00001f;
            float new_learning_rate = abs(learning_rate * loss);
            // Calculate the variable learning rate
            if(nodes_fired.size() == 0)
            {
                cout << "zero fired! \n";
                vector<node *> new_nodes = vs.get_all();
                for (int i = outputs_num; i < new_nodes.size(); i++)
                {
                    vector<node *> last_layer = vs.radius_search_all(new_nodes[i]);
                    vector<float> grad_weight_pos_vec_sum(3, 0.0); // Initialize with zero

                    float node_output = new_nodes[i]->value; // Get the output of the firing node

                    for (node* last_node : last_layer)
                    {
                        float distance = vs.get_distance(new_nodes[i], last_node);
                        if (distance == 0) continue; // Skip zero distance to avoid division by zero

                        float weight = ((1 / (distance)) * new_nodes[i]->s);
                        new_nodes[i]->used = true;

                        // Calculate the gradient vector component-wise
                        vector<float> grad_weight_pos_vec = {
                            loss * new_learning_rate * (-(new_nodes[i]->x - last_node->x) / (distance * distance * distance + 1e-8f)),
                            loss * new_learning_rate * (-(new_nodes[i]->y - last_node->y) / (distance * distance * distance + 1e-8f)),
                            loss * new_learning_rate * (-(new_nodes[i]->z - last_node->z) / (distance * distance * distance + 1e-8f)),
                            loss * new_learning_rate * (-(nodes_fired[i]->bias + last_node->bias))
                        };

                        grad_weight_pos_vec_sum[0] += (grad_weight_pos_vec[0] * not_zero(grad_weight_pos_vec_sum[0]));
                        grad_weight_pos_vec_sum[1] += (grad_weight_pos_vec[1] * not_zero(grad_weight_pos_vec_sum[1]));
                        grad_weight_pos_vec_sum[2] += (grad_weight_pos_vec[2] * not_zero(grad_weight_pos_vec_sum[2]));
                    }
                    if(i % 10 == 0 && verbose)
                    {
                        cout << "grad: [" << grad_weight_pos_vec_sum[0] << "," << grad_weight_pos_vec_sum[2] << "," << grad_weight_pos_vec_sum[3] << "," << "] \n";
                    }
                    
                    // Update biases and weights with the computed gradients
                    new_nodes[i]->bias -= new_learning_rate; // Bias update
                    new_nodes[i]->x -= grad_weight_pos_vec_sum[0] * new_learning_rate;
                    new_nodes[i]->y -= grad_weight_pos_vec_sum[1] * new_learning_rate;
                    new_nodes[i]->z -= grad_weight_pos_vec_sum[2] * new_learning_rate;

                    new_nodes[i]->fired = false;
                    new_nodes[i]->staged = false;
                }
                
                for (int i = 0; i < 2000; i++)
                {
                    vs.add_vector();
                }
                
                back_prop_iteration += 1;
            }
            else
            {
                for (int i = 0; i < nodes_fired.size(); i++)
                {
                    vector<node *> last_layer = vs.radius_search_all(nodes_fired[i]);
                    vector<float> grad_weight_pos_vec_sum(4, 0.0); // Initialize with zero
                    
                
                    for (node* last_node : last_layer)
                    {
                        
                        float distance = vs.get_distance(nodes_fired[i], last_node);
                        if (distance == 0) continue; // Skip zero distance to avoid division by zero
                        float node_output = nodes_fired[i]->value; // Get the output of the firing node
                        if(i < outputs_num)
                        {
                            nodes_fired[i]->bias -= loss * new_learning_rate * (-(nodes_fired[i]->bias + last_node->bias)) * learning_rate; // Bias update
                            nodes_fired[i]->x -= nodes_fired[i]->s * loss * new_learning_rate * (-(nodes_fired[i]->x - last_node->x) / (distance * distance * distance + 1e-8f)),
                            nodes_fired[i]->z -= nodes_fired[i]->s * loss * new_learning_rate * (-(nodes_fired[i]->y - last_node->y) / (distance * distance * distance + 1e-8f)),
                            nodes_fired[i]->z -= nodes_fired[i]->s * loss * new_learning_rate * (-(nodes_fired[i]->z - last_node->z) / (distance * distance * distance + 1e-8f)),

                            nodes_fired[i]->fired = false;
                            nodes_fired[i]->staged = false;
                        }
                        else
                        {
                            vector<float> grad_weight_pos_vec = {
                            nodes_fired[i]->s * loss * new_learning_rate * (-(nodes_fired[i]->x - last_node->x) / (distance * distance * distance + 1e-8f)),
                            nodes_fired[i]->s * loss * new_learning_rate * (-(nodes_fired[i]->y - last_node->y) / (distance * distance * distance + 1e-8f)),
                            nodes_fired[i]->s * loss * new_learning_rate * (-(nodes_fired[i]->z - last_node->z) / (distance * distance * distance + 1e-8f)),
                            loss * new_learning_rate * (-(nodes_fired[i]->bias + last_node->bias))
                            };

                            grad_weight_pos_vec_sum[0] += (grad_weight_pos_vec[0] * not_zero(grad_weight_pos_vec_sum[0]));
                            grad_weight_pos_vec_sum[1] += (grad_weight_pos_vec[1] * not_zero(grad_weight_pos_vec_sum[1]));
                            grad_weight_pos_vec_sum[2] += (grad_weight_pos_vec[2] * not_zero(grad_weight_pos_vec_sum[2]));
                            grad_weight_pos_vec_sum[3] += (grad_weight_pos_vec[3] * not_zero(grad_weight_pos_vec_sum[3]));
                        

                        
                            if(i % 50 == 0 && verbose)
                            {
                                cout << "grad: [" << grad_weight_pos_vec_sum[0] << "," << grad_weight_pos_vec_sum[2] << "," << grad_weight_pos_vec_sum[3] << "," << "] \n";
                            }
                            // Update biases and weights with the computed gradients
                            nodes_fired[i]->bias -= grad_weight_pos_vec_sum[3] * learning_rate; // Bias update
                            nodes_fired[i]->x -= grad_weight_pos_vec_sum[0] * learning_rate;
                            nodes_fired[i]->y -= grad_weight_pos_vec_sum[1] * learning_rate;
                            nodes_fired[i]->z -= grad_weight_pos_vec_sum[2] * learning_rate;

                            nodes_fired[i]->fired = false;
                            nodes_fired[i]->staged = false;
                        }
                        //float weight = ((1 / (distance)) * nodes_fired[i]->s);

                        // Calculate the gradient vector component-wise
                        
                    }
                }
                back_prop_iteration += 1;
                nodes_fired.clear();
            }
            
            if(back_prop_iteration % 40 == 0)
            {
                cout << "loss: " << loss << "\n";
                vs.save();
            }
            vs.re_map();
            if(back_prop_iteration % 100 == 0)
            {
                cout << "loss: " << loss << "\n";
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