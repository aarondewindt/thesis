//
// Created by elvieto on 23-11-18.
//

#ifndef POLE_POLE_H
#define POLE_POLE_H

class Pole{
public:
    double theta;
    double theta_dot;
    double time;
    double dt;
    double mass;
    double length;
    double inertia;

    Pole(double mass, double length);

    /// @param [in] torque Torque applied on the pole.
    /// @param [out] reward Reward from the environment.
    /// @param [out] is_terminal True if the environment is in a terminal state.
    void act(double torque, double &reward, bool &is_terminal);
};


#endif //POLE_POLE_H
