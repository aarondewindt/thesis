//
// Created by adewindt on 6/7/20.
//

#ifndef POLE_NEW_SPANCPY_H
#define POLE_NEW_SPANCPY_H

#include <stdexcept>
#include <cstring>
#include <initializer_list>
#include "types.h"

namespace pole {
    template<class T>
    void spancpy(tcb::span<T> destination, tcb::span<const T> source) {
        if (source.size() > destination.size()) {
            throw std::length_error("Destination smaller than source.");
        }
        std::memcpy(destination.data(), source.data(), source.size_bytes());
    }

    template<class T>
    void spancpy(tcb::span<T> destination, std::initializer_list<T> source){
        spancpy(destination, tcb::span<const T>(source));
    }
}


#endif //POLE_NEW_SPANCPY_H
