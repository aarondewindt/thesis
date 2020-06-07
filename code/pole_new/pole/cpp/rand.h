//
// Created by elvieto on 25-11-18.
//

#ifndef POLE_RAND_H
#define POLE_RAND_H

#include <cstdlib>
#include "types.h"
#include <effolkronium/random.hpp>

namespace pole {
    // get base random alias which is auto seeded and has static API and internal state
    using Random = effolkronium::random_static;

    inline f64 frand(double min, double max)
    {
        return Random::get(min, max);
    }

    inline f64 frand()
    {
        return Random::get(-1.0, 1.0);
    }

    inline usize usize_rand(usize min, usize max) {
        return Random::get(min, max);
    }
}


#endif //POLE_RAND_H
