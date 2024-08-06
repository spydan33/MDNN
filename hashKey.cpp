#include <vector>
#include <cmath>
#include <unordered_map>
#include <iostream>

struct node 
{
    float x, y, z, s;
    bool fired = false;
    bool staged = false;
    bool used = false;
    float value = 0;
    float bias = 0;
    std::tuple< int, int, int, int> to_tuple()
    {
        return {    static_cast<int>(x),
                    static_cast<int>(y),
                    static_cast<int>(z),
                    static_cast<int>(s)
        };
    }
};

// Helper to create hash key from three integers
struct KeyHash 
{
    std::size_t operator()(const std::tuple<int, int, int, int>& k) const 
    {
        // Using std::get to access tuple elements
        auto x = std::get<0>(k);
        auto y = std::get<1>(k);
        auto z = std::get<2>(k);
        auto s = std::get<3>(k);
        return std::hash<int>()(x) ^ std::hash<int>()(y) << 1 ^ std::hash<int>()(z) << 2 ^ std::hash<int>()(s) << 3;
    }
};


// Helper to compare keys for equality in hash map
struct KeyEqual 
{
    bool operator()(const std::tuple<int, int, int, int>& a, const std::tuple<int, int, int,int>& b) const 
    {
        return a == b;
    }
};

