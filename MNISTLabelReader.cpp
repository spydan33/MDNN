#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

class MNISTLabelReader {
public:
    MNISTLabelReader(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Read the magic number
        uint32_t magic = readUInt32(file);
        if (magic != 2049) {
            throw std::runtime_error("Invalid magic number in MNIST label file: " + filename);
        }

        // Read the number of labels
        num_labels = readUInt32(file);

        // Read the label data
        labels.resize(num_labels);
        file.read(reinterpret_cast<char*>(labels.data()), num_labels);
    }

    std::vector<std::vector<float>> getOneHotLabels() const {
        std::vector<std::vector<float>> one_hot_labels(num_labels, std::vector<float>(10, 0.0f));
        for (size_t i = 0; i < num_labels; ++i) {
            one_hot_labels[i][labels[i]] = 1.0f;
        }
        return one_hot_labels;
    }

private:
    uint32_t readUInt32(std::ifstream& file) {
        uint32_t result = 0;
        file.read(reinterpret_cast<char*>(&result), sizeof(result));
        return (result >> 24) | 
               ((result << 8) & 0x00FF0000) |
               ((result >> 8) & 0x0000FF00) |
               (result << 24);
    }

    uint32_t num_labels;
    std::vector<uint8_t> labels;
};
