#pragma once
#include "vector_space.h"
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
        int seed;


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
        MDNN(int n_vect, string filename, int seed) : vs(vector_space(filename,seed)), seed(seed)
        {
            vs.generate_vectors(n_vect);
        };
        MDNN(string filename) : vs(vector_space(filename))
        {
            
        };
        MDNN(vector_space vs_pass,string filename,int seed,int outputs_num,float action_potential_threshold,float learning_rate) : vs(vs_pass), seed(seed), outputs_num(outputs_num),action_potential_threshold(action_potential_threshold),learning_rate(learning_rate)
        {
            vs.filename = filename;
        };
        MDNN mutate(string filename,float mutation_rate)
        {
            auto vs_temp = vs.mutate(mutation_rate);
            return MDNN(vs_temp,filename,seed,outputs_num,action_potential_threshold,learning_rate);
        }
        vector_space* get_vs_pointer()
        {
            vector_space* vs_pointer = &vs;
            return vs_pointer;
        };
        string get_filename()
        {
            return vs.filename;
        }
        void reset()
        {
            output_nodes.clear();
            nodes_fired.clear();
            output.clear();
            vs.reset();
        };
        vector<float>* get_output() 
        {
            return &output;
        }
        vector<float> cascade(vector<float> input)
        {
            try
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
                if(verbose)
                {
                    cout << "Next layer size: ";
                    cout << fireing_nodes.size() << "\n";
                    cout << "finished first forword \n";
                }
                return cascade(fireing_nodes);
            }
            catch (const std::exception &e)
            {
                cout << "An error occurred: " << e.what() << endl;
            }
        }
    
        vector<float> cascade(vector<node*> &input)
        {
            if(verbose)
            {
                cout << "next layer hit \n";
            }
            try
            {
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
                    if(verbose)
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
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
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
                        //float gradient_scalar = -2.0f * loss * firing_node->value / (pow(distance, 3) + 1e-8f); by chat gpt
                        
                        vector<float> grad_weight_pos_vec = {
                            loss * new_learning_rate * (-(new_nodes[i]->x - last_node->x) / (distance * distance * distance + 1e-8f)),
                            loss * new_learning_rate * (-(new_nodes[i]->y - last_node->y) / (distance * distance * distance + 1e-8f)),
                            loss * new_learning_rate * (-(new_nodes[i]->z - last_node->z) / (distance * distance * distance + 1e-8f)),
                            loss * new_learning_rate * (-(nodes_fired[i]->bias + last_node->bias))
                        };
                        /*vector<float> grad_weight_pos_vec = { by chatgpt
                            gradient_scalar * (new_nodes[i]->x - last_node->x),
                            gradient_scalar * (new_nodes[i]->y - last_node->y),
                            gradient_scalar * (new_nodes[i]->z - last_node->z),
                            -2.0f * error * learning_rate * (nodes_fired[i]->bias + last_node->bias) // bias term
                        };*/

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
                vs.prune_file();
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
        int prune()
        {
           auto n_nodes = vs.prune();
           return n_nodes;
        }
        void kill()
        {
            vs.empty();
            output_nodes.clear();
            output.clear();
            nodes_fired.clear();
        }
        void check_used()
        {
            auto nodes = vs.get_all();
            int count = 0;
            int count_2 = 0;
            for(node* t_node : nodes)
            {
                if(t_node->used)
                {
                    count ++;
                }
                if(t_node->fired)
                {
                    count_2 ++;
                }
            }
            cout << "used from vs : " << count << "\n"; 
            cout << "fired from vs : " << count_2 << "\n"; 
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
        void save()
        {
            vs.save();
        }

};