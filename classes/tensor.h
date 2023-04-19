#include <vector>
#include <stdexcept>

/*
you can pass a Tensor object to a function expecting a 2D vector type by using the 
address-of operator (&) to pass a reference to the underlying vector data
*/

template <typename T>
class Tensor {
public:
    Tensor(int size1) : data(size1) {}
    Tensor(int size1, int size2) : data(size1, std::vector<T>(size2)), size1(size1), size2(size2) {}
    Tensor(int size1, int size2, int size3) : data(size1, std::vector<T>(size2, std::vector<T>(size3))), size1(size1), size2(size2), size3(size3) {}

    std::vector<T>& operator[](int i) {
        if (size1 == 0) {
            throw std::out_of_range("Tensor is not a 1D vector");
        }
        return data[i];
    }

private:
    std::vector<std::vector<std::vector<T>>> data;
    int size1 = 0;
    int size2 = 0;
    int size3 = 0;
};

/*
Tensor<int> td(3, 4);
td[1][2] = 99;
*/

// template <typename T>
// class Tensor {
// public:
//     Tensor(int size1) : data(size1) {}
//     Tensor(int size1, int size2) : data(size1 * size2), size1(size1), size2(size2) {}
//     Tensor(int size1, int size2, int size3) : data(size1 * size2 * size3), size1(size1), size2(size2), size3(size3) {}

//     T& operator[](int i) {
//         if (size1 == 0) {
//             throw std::out_of_range("Tensor is not a 1D vector");
//         }
//         return data[i];
//     }

//     T& operator()(int i, int j) {
//         if (size2 == 0) {
//             throw std::out_of_range("Tensor is not a 2D vector");
//         }
//         return data[i * size2 + j];
//     }

//     T& operator()(int i, int j, int k) {
//         if (size3 == 0) {
//             throw std::out_of_range("Tensor is not a 3D vector");
//         }
//         return data[i * size2 * size3 + j * size3 + k];
//     }

// private:
//     std::vector<std::vector<std::vector<T>>> data;
//     int size1 = 0;
//     int size2 = 0;
//     int size3 = 0;
// };

/*

#include <iostream>

int main() {
    Tensor<int> t1d(10);
    t1d[0] = 42;
    std::cout << t1d[0] << std::endl;

    Tensor<int> t2d(3, 4);
    t2d(1, 2) = 99;
    std::cout << t2d(1, 2) << std::endl;

    Tensor<int> t3d(2, 3, 4);
    t3d(0, 1, 2) = 123;
    std::cout << t3d(0, 1, 2) << std::endl;

    // Out of range error
    //t1d(0, 0);
    //t2d(0, 0, 0);

    return 0;
}




Tensor<int> td(3, 4);
td[1][2] = 99;


*/