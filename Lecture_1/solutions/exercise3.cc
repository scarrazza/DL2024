#include <iostream>

int main()
{
    int a = 1, b = 1;
    int target = 48;
    for(int n = 3; n <= target; ++n)
    {
        int fib = a + b;
        std::cout << "F("<< n << ") = " << fib << std::endl;
        a = b;
        b = fib;
    }

    return 0;
}
