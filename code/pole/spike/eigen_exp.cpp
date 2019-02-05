//
// Created by elvieto on 25-11-18.
//

#include <iostream>
#include <Eigen/Dense>

const size_t prl_len = 27;
typedef Eigen::Matrix<double, prl_len, 1> PolyRLColVector;

PolyRLColVector get_x(double s1, double s2, double a);

int main()
{
    PolyRLColVector x = get_x(1, 2, 3);
    std::cout << x << std::endl;
}




PolyRLColVector get_x(double s1, double s2, double a) {
    PolyRLColVector x;
    x <<
      1,
            a,
            a*a,
            s2,
            a*s2,
            a*a*s2,
            s2*s2,
            a*s2*s2,
            a*a*s2*s2,
            s1,
            a*s1,
            a*a*s1,
            s1*s2,
            a*s1*s2,
            a*a*s1*s2,
            s1*s2*s2,
            a*s1*s2*s2,
            a*a*s1*s2*s2,
            s1*s1,
            a*s1*s1,
            a*a*s1*s1,
            s1*s1*s2,
            a*s1*s1*s2,
            a*a*s1*s1*s2,
            s1*s1*s2*s2,
            a*s1*s1*s2*s2,
            a*a*s1*s1*s2*s2
            ;
    return x;
}
