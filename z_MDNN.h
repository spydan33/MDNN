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
        void set_learning_rate(float lr)
        {
            learning_rate = lr;
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
        void back_propagation(std::vector<float> expected_output)
        {
            if (output_nodes.size() != expected_output.size()) return;

            constexpr float kEps = 1e-6f;
            const float invN2 = 2.0f / static_cast<float>(expected_output.size());
            float loss = calc_loss(expected_output);

            // ---- 1) Compute deltas (δ) ----
            std::unordered_map<node*, float> delta;

            // Outputs
            for (size_t i = 0; i < output_nodes.size(); ++i)
            {
                node* out = output_nodes[i];
                float e   = (output[i] - expected_output[i]);
                float d   = invN2 * e;  // linear output
                // If not linear, multiply by activation derivative at z_out:
                // d *= out->value * (1.0f - out->value);          // sigmoid
                // d *= (1.0f - out->value * out->value);          // tanh/normalize
                // d *= (out->value > 0.0f ? 1.0f : 0.0f);         // relu
                delta[out] = d;
            }

            // Hidden fired nodes: δ_j = f'(z_j) * sum_k δ_k * w_{k<-j}
            // Iterate reverse to propagate from outputs back through the fired path.
            for (int idx = static_cast<int>(nodes_fired.size()) - 1; idx >= 0; --idx)
            {
                node* j = nodes_fired[idx];
                if (delta.count(j)) continue; // already an output

                float accum = 0.0f;

                // "Children" k: nodes that could have used j as an input (local neighborhood)
                auto nexts = vs.radius_search_all(j);
                for (node* k : nexts)
                {
                    if (!delta.count(k)) continue; // only those already with δ (downstream)
                    float dkj = vs.get_distance(k, j);
                    if (dkj < kEps) continue;

                    // w_{k<-j} = s_k / d(k,j)
                    accum += delta[k] * (k->s / dkj);
                }

                // Activation derivative at j (assume linear unless you know otherwise)
                float d = accum;
                // d *= j->value * (1.0f - j->value);           // sigmoid
                // d *= (1.0f - j->value * j->value);           // tanh/normalize
                // d *= (j->value > 0.0f ? 1.0f : 0.0f);        // relu

                // Only keep meaningful deltas to avoid tiny noise
                if (std::isfinite(d) && std::fabs(d) > 0.0f) delta[j] = d;
            }

            // ---- 2) Apply parameter updates (bias + position) to all nodes with δ ----
            for (auto &kv : delta)
            {
                node* j   = kv.first;
                float dj  = kv.second;

                // Bias
                j->bias -= learning_rate * dj;

                // Position gradient has two parts:
                // (A) Incoming edges into j   (depends on j via w_{j<-i} = s_j / d(j,i))
                // (B) Outgoing edges to k     (depends on j via w_{k<-j} = s_k / d(k,j))

                float gx = 0.0f, gy = 0.0f, gz = 0.0f;

                // (A) Incoming: ∂L/∂x_j += δ_j * sum_i y_i * s_j * (+ r_ji / d^3)
                {
                    auto incomers = vs.radius_search_all(j);
                    for (node* i : incomers)
                    {
                        float dji = vs.get_distance(j, i);
                        if (dji < kEps) continue;
                        const float invd3 = 1.0f / (dji * dji * dji);
                        const float coeff = dj * j->s * i->value * invd3; // NOTE: + sign
                        gx += coeff * (j->x - i->x);
                        gy += coeff * (j->y - i->y);
                        gz += coeff * (j->z - i->z);
                    }
                }

                // (B) Outgoing: ∂L/∂x_j += sum_k δ_k * y_j * s_k * (+ r_kj / d^3)
                {
                    auto children = vs.radius_search_all(j);
                    for (node* k : children)
                    {
                        if (!delta.count(k)) continue; // only those carrying error downstream
                        float dkj = vs.get_distance(k, j);
                        if (dkj < kEps) continue;
                        const float invd3 = 1.0f / (dkj * dkj * dkj);
                        const float coeff = delta[k] * j->value * k->s * invd3; // NOTE: + sign
                        gx += coeff * (k->x - j->x);
                        gy += coeff * (k->y - j->y);
                        gz += coeff * (k->z - j->z);
                    }
                }

                j->x += learning_rate * gx;
                j->y += learning_rate * gy;
                j->z += learning_rate * gz;

                j->fired  = false;
                j->staged = false;
            }

            nodes_fired.clear();
            back_prop_iteration += 1;

            if (back_prop_iteration % 40 == 0) { std::cout << "loss: " << loss << "\n"; vs.save(); }
            vs.re_map();
            if (back_prop_iteration % 100 == 0) { std::cout << "loss: " << loss << "\n"; vs.prune(); }
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