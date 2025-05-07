#pragma once
#include "z_MDNN.h"
#include "MNISTImageReader.h"
#include "MNISTLabelReader.h"
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <exception>
#include <iostream>
using namespace std;

class network
{
    public:
        int node_count;
        int generation_id;
        int population_id;
        vector<float> error;
        int seed;
        MDNN nn;
        float avg_error;
        double average_speed;
        double fitness;

        network()
            : generation_id(0), population_id(0), node_count(1000), nn(3000, "vectorspace_default",seed),seed(seed)
        {}

        network(int generation_id, int population_id, int seed)
            : generation_id(generation_id),
            population_id(population_id),
            node_count(1000),
            nn(1000, "vectorspace_" + to_string(generation_id) + "_" + to_string(population_id), seed)
        {}

        network(int node_count)
            : generation_id(0), population_id(0), node_count(node_count), nn(1000, "vectorspace_default",seed),seed(seed)
        {}

        network(int generation_id, int population_id, int seed, int node_count)
            : generation_id(generation_id),
            population_id(population_id),
            node_count(node_count),
            nn(node_count, "vectorspace_" + to_string(generation_id) + "_" + to_string(population_id), seed)
        {}

        network(MDNN nn_pass,int node_count,int generation_id, int population_id, int seed)//mutate constructor
            : 
            nn(nn_pass),
            node_count(node_count),
            generation_id(generation_id),
            population_id(population_id),
            seed(seed)
        {}

        void kill()
        {
            nn.kill();
        }

        void calculate_average_error()
        {
            float error_sum = 0;
            for(float _error : error)
            {
                error_sum += _error;
            }
            avg_error = error_sum / error.size();
        }
        network mutate(int generation_id,int population_id,float mutation_rate)
        {
            MDNN nn_temp = nn.mutate("vectorspace_" + to_string(generation_id) + "_" + to_string(population_id),mutation_rate);
            return network(nn_temp,node_count,generation_id,population_id,seed);
        }
};

class train
{
    private:
        vector<network*> generation;
        std::random_device rd;
        std::mt19937 master_gen = std::mt19937(rd());

    int get_random_number(float mean, float stddev) 
    {
        // Normal distribution generator
        std::normal_distribution<> distr(mean, stddev); // Define the mean and standard deviation
        return static_cast<int>(distr(master_gen)); // Generate and return the random number
    }
    public:
        int population_size;
        int num_generations;
        float mutation_rate;
        int population_survival_count;
        int itteration_per_population;
        bool proceed_by_step = false;
        int image_itt = 0;

    train()
        : population_size(10), num_generations(5), mutation_rate(0.1f), population_survival_count(10),itteration_per_population(100)
    {
        cout << "new train \n";
    }
    void run()
    {
        try
        {
            MNISTImageReader reader("train-images.idx3-ubyte");
            const auto &images = reader.getImages();
            MNISTLabelReader reader_1("train-labels.idx1-ubyte");
            auto one_hot_labels = reader_1.getOneHotLabels();
            for (int gen = 0; gen < num_generations; gen++)
            {
                cout << "\r generation: " << gen << "\n" << flush;
                for (int pop = 0; pop < population_size; pop++)
                {
                    if(gen == 0)
                    {
                        network* species = new network(gen,pop,master_gen(),get_random_number(1000,200));
                        generation.push_back(species);
                    }
                    else
                    {
                        int current_population = generation.size();
                        while (generation.size() < population_size)
                        {
                            int parent_index = (generation.size() - current_population) % current_population;
                            network* parent = generation[parent_index];
                            network child = parent->mutate(gen + 1, generation.size(), mutation_rate);
                            generation.push_back(new network(child));
                        }
                    }
                }
                cout << "Generation " << gen << " complete\n" << flush;
                for(int i = 1; i < images.size(); i ++)
                {
                    image_itt ++;
                    if(i == itteration_per_population)
                    {
                        break;
                    }
                    //cout << "\r image: " << i << "\n" << flush;
                    for(int ii = 1; ii < generation.size(); ii++ )
                    {
                        network* species = generation[ii];
                        auto ret = species->nn.cascade(images[image_itt].pixels);
                        species->error.push_back(species->nn.calc_loss(one_hot_labels[image_itt]));
                        species->nn.reset();
                    }
                }
                for(network* species : generation)
                {
                    species->calculate_average_error();
                    species->error.clear();
                    species->node_count = species->nn.prune();
                }
                generation.erase(
                std::remove_if(generation.begin(), generation.end(),
                [](network* species)
                {
                    return std::isnan(species->avg_error) || std::isinf(species->avg_error);
                }),
                generation.end());
                std::sort(generation.begin(), generation.end(),
                [](network* a, network* b)
                {
                    return a->avg_error < b->avg_error;
                });
                while (generation.size() > population_survival_count)
                {
                    network *temp = generation.back();
                    temp->kill();
                    delete temp;
                    generation.pop_back();
                }
                for(network* species : generation)
                {
                    cout << "Survival Species " << species->population_id << ": average Error: " << species->avg_error << " node count: "<< species->node_count << endl;
                    species->nn.check_used();
                    if(gen == (num_generations - 1))
                    {
                        species->nn.save();
                    }
                }
            }
        }
        catch (const std::exception &e)
        {
            cout << "An error occurred: " << e.what() << endl;
        }
    }
};