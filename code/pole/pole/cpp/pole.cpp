//
// Created by elvieto on 23-11-18.
//

#include "pole.h"
#include <cmath>

#define MAX_THETA 3.141592653589793 // 0.523598776  // 30 degrees

const double pi = 3.141592653589793;
const double pi2 = 6.283185307179586;

Pole::Pole(double mass, double length):
        theta(0),
        theta_dot(0),
        time(0),
        dt(0.01),
        length(length),
        mass(mass),
        inertia(mass * length * length / 3){

}

void Pole::act(double torque, double &reward, bool &is_terminal) {
//    torque = 0;
    torque += mass * 9.81 * (length / 2) * sin(theta);
    // Rotational acceleration
    double theta_dot_dot = torque / inertia;

    // Integrate
    theta_dot += theta_dot_dot * dt;
    theta += theta_dot * dt;
    time += dt;

    theta = fmod(fmod(theta + pi, pi2) + pi2, pi2) - pi;

    reward = pow((MAX_THETA - fabs(theta)) / MAX_THETA, 4) - 0.2;
    is_terminal = false;

//    if (fabs(theta) > MAX_THETA) {
//        reward = -1.0;
//        is_terminal = true;
//    } else {
//        reward = pow((MAX_THETA - fabs(theta)) / MAX_THETA, 16) - 0.2;
//        is_terminal = false;
//    }
}
