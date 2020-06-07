//
// Created by adewindt on 6/7/10.
//

#ifndef POLE_NEW_TABLE_3D_H
#define POLE_NEW_TABLE_3D_H

#include "types.h"
#include "tcb/span.hpp"


namespace pole {
    template<class T, class Y>
    class Table3D {
        T* data;
        Y min0;
        Y max0;
        Y delta0;
        usize n0;

        Y min1;
        Y max1;
        Y delta1;
        usize n1;

        Y min2;
        Y max2;
        Y delta2;
        usize n2;

        usize c1;
        usize c2;

    public:
        Table3D(Y min0, Y max0, usize n0,
                Y min1, Y max1, usize n1,
                Y min2, Y max2, usize n2) :
                min0(min0), max0(max0), n0(n0),
                min1(min1), max1(max1), n1(n1),
                min2(min2), max2(max2), n2(n2),
                c1(n0), c2(n0 * n1),
                delta0((max0 - min0) / n0),
                delta1((max1 - min1) / n1),
                delta2((max2 - min2) / n2) {
            data = new T[n0 * n1 * n2];
        }

        ~Table3D() {
            delete[] data;
        }

        tcb::span<T> get_span() {
            return tcb::span<T>(data, n0 * n1 * n2);
        }

        T& operator()(Y x0, Y x1, Y x2){
            auto idx0 = static_cast<usize>((x0 - min0) / delta0);
            auto idx1 = static_cast<usize>((x1 - min1) / delta1);
            auto idx2 = static_cast<usize>((x2 - min2) / delta2);
            return get(idx0, idx1, idx2);
        }
        
        T& get(usize idx0, usize idx1, usize idx2) {
            if (idx0 >= n0) idx0 = n0 - 1;
            if (idx1 >= n1) idx1 = n1 - 1;
            if (idx2 >= n2) idx2 = n2 - 1;
            usize idx = idx0 + c1 * idx1 + c2 * idx2;
            return data[idx];
        }

        tcb::span<T> operator()(Y x1, Y x2){
            auto idx1 = static_cast<usize>((x1 - min1) / delta1);
            auto idx2 = static_cast<usize>((x2 - min2) / delta2);
            return get(idx1, idx2);
        }

        tcb::span<T> get(usize idx1, usize idx2) {
            if (idx1 >= n1) idx1 = n1 - 1;
            if (idx2 >= n2) idx2 = n2 - 1;
            usize idx = c1 * idx1 + c2 * idx2;
            return tcb::span(data + idx, n0);
        }
    };
}

#endif //POLE_NEW_TABLE_2D_H
