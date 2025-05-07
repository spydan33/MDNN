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

class run
{
    private:
        MDNN* nn;
        int images_count;
        int images_start;
    void print_output(vector<float> vec)
    {
        cout << "[";
        for(float output : vec)
        {
            cout << output <<",";
        }
        cout << "] \n";
    }
    public:
    run(int i_images,int i_count, MDNN* nn_pass): images_start(i_images),images_count(i_count),nn(nn_pass)
    {

    }
    void test()
    {
        try
        {
            cout << "Testing...\n";
            MNISTImageReader reader("train-images.idx3-ubyte");
            const auto &images = reader.getImages();
            MNISTLabelReader reader_1("train-labels.idx1-ubyte");
            auto one_hot_labels = reader_1.getOneHotLabels();
            for(int i = images_start; i < images.size(); i ++)
            {
                if(i == images_count)
                {
                    break;
                }
                cout << "------------image " << i << " -----------------\n" ;
                auto ret = nn->cascade(images[i].pixels);
                cout << "Network Guess \n";
                print_output(ret);
                cout << "True Answer \n";
                print_output(one_hot_labels[i]);
                cout << "------------------------\n";
                nn->reset();
                cout << "Enter anything to continue... \n";
                string in;
                cin >> in; 
            }
        }
        catch (const std::exception &e)
        {
            cout << "An error occurred: " << e.what() << endl;
        }
    }
};