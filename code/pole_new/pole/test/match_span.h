//
// Created by adewindt on 6/7/20.
//

#ifndef MATCH_SPAN_ARRAY_H
#define MATCH_SPAN_ARRAY_H

#include "catch2/catch.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>

#include "tcb/span.hpp"

template <class T>
class _SpanEquals : public Catch::MatcherBase<tcb::span<T>> {
    tcb::span<const T> correct;

public:
    explicit _SpanEquals(tcb::span<const T> correct) :
            correct(correct) {}

    // Match for number of T type.
    bool match(tcb::span<T> const& real) const override {
        // Fail if the sizes are not the same.
        if (real.size() != correct.size()) {
            return false;
        }

        // Iterate through each element and check they are
        // equal. Fail if they are not the same.
        for (size_t i = 0; i < correct.size(); i++) {
            if (real[i] != correct[i]) {
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] std::string describe() const override {
        std::ostringstream ss;
        ss << "span array equal to ";
        for (auto value : correct) {
            ss << value << " ";
        }
        return ss.str();
    }
};

template <class T>
_SpanEquals<const T> SpanEquals(tcb::span<const T> correct) {
    return _SpanEquals<const T>(correct);
}

template <class T>
_SpanEquals<const T> SpanEquals(std::initializer_list<T> correct) {
    tcb::span<const T> v(correct);
    return _SpanEquals<const T>(v);
}

template <class T>
_SpanEquals<const T> SpanEquals(const char *correct) {
    return _SpanEquals<const T>(tcb::span<const T>(reinterpret_cast<const T*>(correct), strlen(correct) / sizeof(T)));
}


template <class T, class Y>
class _SpanApprox : public Catch::MatcherBase<tcb::span<T>> {
    tcb::span<const T> correct;
    const Y relative_tolerance;
    const Y absolute_tolerance;

public:
    explicit _SpanApprox(tcb::span<const T> correct, Y relative_tolerance, Y absolute_tolerance) :
            correct(correct), relative_tolerance(relative_tolerance), absolute_tolerance(absolute_tolerance) {}

    // Match for number of T type.
    bool match(tcb::span<T> const& real) const override {
        // Fail if the sizes are not the same.
        if (real.size() != correct.size()) {
            return false;
        }

        // Iterate through each element and check they are
        // equal. Fail if they are not the same.
        for (size_t i = 0; i < correct.size(); i++) {
            Y a = real[i];
            Y b = correct[i];
            if (!(fabs(a - b) <= (absolute_tolerance + relative_tolerance * fabs(b)))) {
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] std::string describe() const override {
        std::ostringstream ss;
        ss << "span array approximately equal to ";
        for (auto value : correct) {
            ss << value << " ";
        }
        return ss.str();
    }
};

template <class T, class Y=double>
_SpanApprox<T, Y> SpanApprox(tcb::span<const T> correct, Y relative_tolerance=1e-8, Y absolute_tolerance=0.0) {
    return _SpanApprox<T, Y>(correct, relative_tolerance, absolute_tolerance);
}

template <class T, class Y=double>
_SpanApprox<T, Y> SpanApprox(std::initializer_list<T> correct, Y relative_tolerance=1e-8, Y absolute_tolerance=0.0) {
    tcb::span<const T> v(correct);
    return _SpanApprox<T, Y>(v, relative_tolerance, absolute_tolerance);
}


template <class T>
class _SpanAllNaN : public Catch::MatcherBase<tcb::span<T>> {
public:
    bool match(tcb::span<T> const& real) const override {
        for (auto value : real) {
            if (!std::isnan(value)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::string describe() const override {
        std::ostringstream ss;
        ss << "all elements in the span are NaN";
        return ss.str();
    }
};

template <class T>
_SpanAllNaN<T> SpanAllNaN() {
    return _SpanAllNaN<T>();
}

#endif // MATCH_SPAN_ARRAY_H
