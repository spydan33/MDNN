# Multi-Dimensional Neural Network (MDNN)

**MDNN** is an experimental C++ neural-network architecture where every neuron is a point in 3-D space.
Connection weights are defined as **1 / distance** between neurons, so physically closer neurons interact more strongly.
By restricting computation to neighbours within a configurable radius, MDNN aims to:

* **Lower computational cost** – skip distant pairs entirely.
* **Reduce memory footprint** – store or load only active clusters.
* **Encourage locality** – specialised subnetworks form naturally.

---

## ✨ Current Feature Set

| Status | Module                     | Notes                                                  |
| :----: | -------------------------- | ------------------------------------------------------ |
|    ✅   | **Vector-Space Grid**      | Spatial hash with radius & k-nearest search            |
|    ✅   | **Forward Cascade**        | Propagates inputs through fired neurons only           |
|    ✅   | **Back-prop Skeleton**     | Gradients flow along the same fired path               |
|    ✅   | **Evolution Engine**       | `network` class supports generations & mutation copies |
|   🛠️  | **Neuron-Wiggle**          | Fine-tunes only recently-fired neurons *(in progress)* |
|   🛠️  | **Smart-Pointer Refactor** | Replacing raw pointers with `std::unique_ptr`          |
|   🛠️  | **Error Convergence**      | Training currently plateaus – needs tuning             |

---

## 🔧 Build Instructions

* **Compiler** : *GNU Compiler Collection* **(GCC)** C++ frontend
* **Standard** : C++20
* **Dependencies** : none (uses only the C++ Standard Library)

```bash
# from project root
g++ -std=c++20 -O3 *.cpp -o mdnn
```

> Using MSVC or Clang? Enable full C++20 support and `std::chrono` extensions.

---

## 🚀 Quick Start

```cpp
#include "MDNN.h"

int main() {
    MDNN nn;

    std::vector<float> input = /* your normalised sample */;
    nn.cascade(input);            // forward pass

    nn.reset();            // reset fired flags
}
```

### Compile-time Constants

| Constant              | Location         | Purpose                  |
| --------------------- | ---------------- | ------------------------ |
| `inputs`              | `MDNN.cpp`       | Total input nodes        |
| `CELL_SIZE`, `RADIUS`, `vector_space_range` | `vector_space.h` | Spatial grid granularity |

---

## 🗄️ Repository Layout

```
z_MDNN.h           core network logic
vector_space.h     spatial hash + distance helpers
train.h            GA wrapper (generations / mutation)
main.cpp           CLI front-end: “train -g 1000 -p 500 …”
run.h              Testing wrapper for trained networks.
hashKey.h          128-bit key for grid buckets and node struct.
README.md          you are here
```

---

## 📈 Project Status <small>(7 May 2025)</small>

* Will progress through generations reducing error.
* Error reduces but output is not changing over varied inputs.

---

## 🛣️ Roadmap

1. Finalise **Proper Cascade** evolutionary fine-tuning.
2. Optimize GA training.
3. Refactor pointers.
4. Benchmark against dense MLP on MNIST & CIFAR-10.
5. Optional Python bindings via **pybind11** for rapid experimentation.

---

## 🤝 Contributing

Issues, pull requests, and design critiques are welcome!
For significant changes, please open an issue first to discuss your ideas.

---

## 📜 License

Released under the **MIT License** – see `LICENSE` for details.

---

## ✉️ Contact

Daniel Mergenthal • open an issue or ping **@Dan-MDNN** on GitHub.
