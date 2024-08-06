#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm> // For std::transform

struct MNISTImage {
    std::vector<float> pixels;
};

class MNISTImageReader {
public:
    MNISTImageReader(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Read the magic number
        uint32_t magic = readUInt32(file);
        if (magic != 2051) {
            throw std::runtime_error("Invalid magic number in MNIST image file: " + filename);
        }

        // Read the number of images, rows, and columns
        num_images = readUInt32(file);
        num_rows = readUInt32(file);
        num_cols = readUInt32(file);

        // Read the image data
        images.resize(num_images);
        for (uint32_t i = 0; i < num_images; ++i) {
            std::vector<uint8_t> temp_pixels(num_rows * num_cols);
            file.read(reinterpret_cast<char*>(temp_pixels.data()), num_rows * num_cols);
            images[i].pixels.resize(num_rows * num_cols);

            // Convert uint8_t pixels to float and normalize to [0, 1]
            std::transform(temp_pixels.begin(), temp_pixels.end(), images[i].pixels.begin(), [](uint8_t pixel) {
                return static_cast<float>(pixel) / 255.0f;
            });
        }
    }

    const std::vector<MNISTImage>& getImages() const {
        return images;
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

    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_cols;
    std::vector<MNISTImage> images;
};


