//
// Created by elvieto on 3-4-19.
//

#ifndef F16_MATCH_POINTER_ARRAY_H
#define F16_MATCH_POINTER_ARRAY_H

#include "catch.hpp"
#include <vector>
#include <cmath>
#include <sstream>


template <class T>
class _PArrayApprox : public Catch::MatcherBase<T*> {
    std::vector<T> correct;
    double relative_tolerance;
    double absolute_tolerance;

public:
    _PArrayApprox(
            std::vector<T> &correct,
            T relative_tolerance,
            T absolute_tolerance) :
                correct( correct ),
                relative_tolerance(relative_tolerance),
                absolute_tolerance(absolute_tolerance) {}

    // Performs the test for this matcher
    bool match( T* const& real ) const override {
        bool pass = true;
        // Iterate from 0 to the last index in the correct vector.
        for (int i = 0; i < correct.size(); i++){
            // Calculate the absolute error.
            double error = fabs(real[i] - correct[i]);

            if (correct[i] == 0) {
                // Only check the absolute tolerance if the correct value is 0.
                pass = pass && (error <= absolute_tolerance);
            } else {
                // Otherwise check both the relative and absolute tolerance.
                pass = pass && ((error / correct[i]) <= relative_tolerance) && (error <= absolute_tolerance);
            }
        }
        return pass;
    }

    std::string describe() const override {
        std::ostringstream ss;
        ss << "pointer array approximately equal to ";
        for (auto value : correct) {
            ss << value << " ";
        }
        return ss.str();
    }
};

template <class T>
inline _PArrayApprox<T> PArrayApprox(std::initializer_list<T> correct) {
    std::vector<T> v(correct);
    return _PArrayApprox<T>(v, 1e-20, 0.0);
}

template <class T>
inline _PArrayApprox<T> PArrayApprox(std::initializer_list<T> correct, T relative_tolerance,
                                     T absolute_tolerance) {
    std::vector<T> v(correct);
    return _PArrayApprox<T>(v, relative_tolerance, absolute_tolerance);
}

#endif //F16_MATCH_POINTER_ARRAY_H
