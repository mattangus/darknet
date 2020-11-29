#include <iostream>

template<int T>
int function()
{
    switch (T)
    {
    case 1:
        return 10;
    case 2:
        return 20;
    case 3:
        return 30;
    case 4:
        return 40;
    }
    return -1;
}

int main()
{
    function<1>();
}