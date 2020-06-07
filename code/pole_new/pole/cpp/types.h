//
// Created by adewindt on 6/7/20.
//

#ifndef POLE_NEW_TYPES_H
#define POLE_NEW_TYPES_H

#include <cstdint>
#include "tcb/span.hpp"

namespace pole {
    using u8 = std::uint8_t;
    using u16 = std::uint16_t;
    using u32 = std::uint32_t;
    using u64 = std::uint64_t;

    using i8 = std::int8_t;
    using i16 = std::int16_t;
    using i32 = std::int32_t;
    using i64 = std::int64_t;

    using usize = std::size_t;
    using isize = std::ptrdiff_t;

    using f32 = float;
    using f64 = double;

    const double pi = 3.141592653589793;
    const double pi2 = 6.283185307179586;
}

#endif //POLE_NEW_TYPES_H
