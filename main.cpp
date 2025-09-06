#include <iostream>
#include "z_MDNN.h"
#include "train.h"
#include "run.h"
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;

struct r_data
{
    MDNN* nn;
    bool loaded = false;
};
// Forward declarations for command handlers
void handleRun(r_data run_data,const vector<string>& args);
void handleTrain(const vector<string>& args);
void handleVectors(const vector<string>& args);
void handleSettings(const vector<string>& args);
MDNN* handleLoad(const vector<string>& args);
void showHelp(const vector<string>& args);

vector<string> splitInput(const string& input)
{
    stringstream ss(input);
    string word;
    vector<string> tokens;
    while (ss >> word)
    {
        tokens.push_back(word);
    }
    return tokens;
}

int main()
{
    try
    {
        string input;
        bool running = true;
        r_data run_data;

        cout << "---- MDNN started ---- \nType 'help' for options.\n";
        while(running)
        {
            cout << "> ";
            getline(cin, input);
            vector<string> args = splitInput(input);

            if (args.empty()) continue;

            string command = args[0];

            if (command == "help")
            {
                showHelp(args);
            }
            else if (command == "quit")
            {
                cout << "Exiting...\n";
                running = false;
            }
            else if (command == "run")
            {
                handleRun(run_data,args);
            }
            else if (command == "train")
            {
                handleTrain(args);
            }
            else if (command == "vectors")
            {
                //handleVectors(args);
            }
            else if (command == "settings")
            {
                //handleSettings(args);
            }
            else if (command == "load")
            {
               run_data.nn = handleLoad(args);
            }
            else if (command == "clear")
            {
                // Cross-platform clear screen
                #ifdef _WIN32
                    system("cls");
                #else
                    system("clear");
                #endif
            }
            else
            {
                cout << "Unknown command: " << command << "\n";
            }
        }
    }
    catch(const exception& e)
    {
        string in;
        cout << "Error: " << e.what() << '\n';
        cout << "Exiting...Enter any value to quit\n";
        cin >> in;
    }

    return 0;
}

// ----------------------------
// Command Help Logic
// ----------------------------
void showHelp(const vector<string>& args)
{
    if (args.size() == 1)
    {
        cout << "Commands:\n"
            << "  run       - Run inference on the model\n"
            << "  train     - Train the model with genetic algorithm or backprop\n"
            << "  vectors   - Manage vector spaces\n"
            << "  settings  - Configure global options\n"
            << "  load      - Load a saved vector space\n"
            << "  clear     - Clear the screen\n"
            << "  quit      - Exit the program\n"
            << "Type 'help [command]' for more details on a specific command.\n";

    }
    else if (args[1] == "run")
    {
        cout << "Usage: run -s [vector size] -i [inputs] -o [outputs] -d [data URL]\n"
            << "Options:\n"
           //  << "  -s    Vector space size\n"
           //  << "  -i    Number of inputs\n"
          //   << "  -o    Number of outputs\n"
            // << "  -d    URL or path to input data\n";
            << "  -s               Start point in data\n"
            << "  -i               How many images to try\n"
            << "  --test  Test a network on MNIST data\n";
    }
    else if (args[1] == "train")
    {
        cout << "Usage: train -p [population size] -m [mutation rate] -g [generations] -d [training data URL] [--backpropagation]\n"
             << "Options:\n"
             << "  -p               Population size for genetic algorithm\n"
             << "  -m               Mutation rate (e.g., 0.01)\n"
             << "  -g               Number of generations\n"
             << "  -d               URL or path to training data\n"
             << "  -k               Number of population winners to keep per generation\n"
             << "  -i               Number of images to train on per generation\n"
             << "  -n               Network file to load/save for backpropagation\n"
             << "  --backpropagation  Use backpropagation instead of the genetic algorithm\n"
             << "  --standard  use standard for dev and save time\n";
    }
    else if (args[1] == "vectors")
    {
        cout << "Usage: vectors [list|create|delete] [options]\n"
             << "Subcommands:\n"
             << "  list              List all vector spaces\n"
             << "  create -n [name]  Create a new vector space with given name\n"
             << "  delete -n [name]  Delete the specified vector space\n";
    }
    else if (args[1] == "settings")
    {
        cout << "Usage: settings [options]\n"
             << "Options:\n"
             << "  --verbose [on|off]               Enable or disable verbose mode\n"
             << "  --cluster-size [number]          Set neuron cluster size\n"
             << "  --activation [relu|sigmoid|tanh] Choose activation function\n"
             << "  --activation-distance [number]   Set distance threshold for neuron activation\n"
             << "  --prune-timing [number]          Set timing interval for pruning neurons\n";
    }
    else if (args[1] == "load")
    {
        cout << "Usage: load -n [name]\n"
             << "Options:\n"
             << "  -n    Name of the vector space to load into memory\n";
    }
    else
    {
        cout << "Unknown command: " << args[1] << "\n";
    }
}
void handleRun(r_data run_data, const vector<string>& args)
{
    if (!run_data.nn)  // check if vector space is loaded
    {
        throw std::runtime_error("No vector space loaded. Use 'load' command before running.");
    }

    int i_images = -1;
    int i_count = -1;
    bool test = false;

    bool has_i = false;
    bool has_s = false;
    bool has_test = false;

    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == "-i" && i + 1 < args.size())
        {
            i_count = stoi(args[++i]);
            has_i = true;
        }
        else if (args[i] == "-s" && i + 1 < args.size())
        {
            i_images = stoi(args[++i]);
            has_s = true;
        }
        else if (args[i] == "--test")
        {
            test = true;
            has_test = true;
        }
        else
        {
            cout << "Unknown or incomplete argument: " << args[i] << "\n";
        }
    }

    if (!has_i || !has_s || !has_test)
    {
        cout << "Must pass -s, -i, and --test currently \n";
    }

    run runner(i_images,i_count,run_data.nn);

    // Pass args into runner if needed
    // e.g. runner.run(run_data.nn, i_count, i_images, test);

    cout << "Running with:\n";
    cout << "  Start index: " << i_images << "\n";
    cout << "  Image count: " << i_count << "\n";
    cout << "  Test mode: " << (test ? "enabled" : "disabled") << "\n";
    runner.test();

    // Your actual run logic here
}
MDNN* handleLoad(const vector<string>& args)
{
    if (args.size() < 2)
    {
        throw std::runtime_error("Not enough arguments for 'load'. Type 'help load' for usage.");
    }

    string filename = "";

    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == "-n" && i + 1 < args.size())
        {
            filename = args[++i];
        }
        else if (args[i] == "--standard")
        {
            
        }
        else
        {
            cout << "Unknown or incomplete argument: " << args[i] << "\n";
        }
    }
    return new MDNN(filename);
}
void handleTrain(const vector<string>& args)
{
    if (args.size() < 2)
    {
        cout << "Error: Not enough arguments for 'train'. Type 'help train' for usage.\n";
        return;
    }
    train trainer;

    string data_url = "";
    bool use_backpropagation = false;
    string network_file = "";

    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == "-p" && i + 1 < args.size())
        {
            trainer.population_size = stoi(args[++i]);
        }
        else if (args[i] == "-m" && i + 1 < args.size())
        {
            trainer.mutation_rate = stof(args[++i]);
        }
        else if (args[i] == "-g" && i + 1 < args.size())
        {
            trainer.num_generations = stoi(args[++i]);
        }
        else if (args[i] == "-d" && i + 1 < args.size())
        {
            data_url = args[++i]; //not used, currently a place holder
        }
        else if (args[i] == "-k" && i + 1 < args.size())
        {
            trainer.population_survival_count = stoi(args[++i]);
        }
        else if (args[i] == "-i" && i + 1 < args.size())
        {
            trainer.itteration_per_population = stoi(args[++i]);
        }
        else if (args[i] == "-n" && i + 1 < args.size())
        {
            network_file = args[++i];
        }
        else if (args[i] == "--backpropagation")
        {
            use_backpropagation = true;
        }
        else if (args[i] == "--standard")
        {
            trainer.population_size = 10; //not used, same as above
            trainer.num_generations = 3; //not used, same as above
            trainer.population_survival_count = 3; //not used, same as above
            trainer.itteration_per_population = 10; //not used, same as above
        }
        else
        {
            cout << "Unknown or incomplete argument: " << args[i] << "\n";
        }
    }

    cout << "Starting training with settings:\n"
         << "  Population size: " << trainer.population_size << "\n"
         << "  Mutation rate: " << trainer.mutation_rate << "\n"
         << "  Generations: " << trainer.num_generations << "\n"
         << "  Data URL: " << data_url << "\n"
         << "  Using backpropagation: " << (use_backpropagation ? "Yes" : "No") << "\n";

    // Call your training logic here
    if (use_backpropagation)
    {
        MDNN nn = network_file.empty() ? MDNN(1000, "backprop_vectorspace", 0) : MDNN(network_file);
        try
        {
            MNISTImageReader image_reader("train-images.idx3-ubyte");
            const auto &images = image_reader.getImages();
            MNISTLabelReader label_reader("train-labels.idx1-ubyte");
            auto one_hot_labels = label_reader.getOneHotLabels();
            int limit = min<int>(trainer.itteration_per_population, images.size());
            for (int i = 0; i < limit; ++i)
            {
                nn.cascade(images[i].pixels);
                nn.back_propagation(one_hot_labels[i]);
                nn.reset();
            }
            nn.save();
            cout << "Backpropagation training complete.\n";
        }
        catch(const std::exception& e)
        {
            cout << "An error occurred: " << e.what() << "\n";
        }
    }
    else
    {
        trainer.run();
    }
}