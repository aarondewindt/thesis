#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <inttypes.h>

//template<int bits>
//struct rint {typedef (std::conditional< (bits > 32), uint64_t,
//                     (std::conditional< (bits > 16), uint32_t,
//                     (std::conditional< (bits > 8), uint16_t, uint8_t >
//                     ::type)>::type)>::type) type;};

typedef unsigned uint128_t __attribute__ ((mode (TI)));

template <int bits>
struct rint {
    typedef typename std::conditional<(bits > 64), uint128_t,
            typename std::conditional<(bits > 32), uint64_t,
            typename std::conditional<(bits > 16), uint32_t,
            typename std::conditional<(bits > 8), uint16_t, uint8_t
    >::type>::type>::type>::type type;
};


int main()
{
//    std::cout << typeid(rint<40>::type).name() << " " << sizeof(rint<40>::type) << '\n';
//    std::cout << typeid(rint<30>::type).name() << " " << sizeof(rint<30>::type) << '\n';
//    std::cout << typeid(rint<10>::type).name() << " " << sizeof(rint<10>::type) << '\n';
//    std::cout << typeid(rint<5>::type).name() << " " << sizeof(rint<5>::type) << '\n';

    std::cout << typeid(rint<70>::type).name() << " " << sizeof(rint<70>::type) << '\n';
    std::cout << typeid(rint<40>::type).name() << " " << sizeof(rint<40>::type) << '\n';
    std::cout << typeid(rint<30>::type).name() << " " << sizeof(rint<30>::type) << '\n';
    std::cout << typeid(rint<10>::type).name() << " " << sizeof(rint<10>::type) << '\n';
    std::cout << typeid(rint<5>::type).name() << " " << sizeof(rint<5>::type) << '\n';
}