#include <iostream>
#include "add.h"
#include <Eigen/Dense>

int main()
{
    std::cout << "Enter two integers: ";
    int x, y;
    std::cin >> x >> y;

    std::cout << "Sum of integers: " << add(x, y) << std::endl; 

    return 0;
}