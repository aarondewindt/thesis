%================================================
%     Matlab Script File used to linearize the 
%     non-linear F-16 model. The program will 
%     Extract the longitudal and lateral 
%     direction matrices.  These system matrices 
%     will be used to create pole-zero mapping
%     and the bode plots of each to each control
%     input.
% Author: Richard S. Russell
% 
%================================================

% Changes from original:
%   1. Removed code for the hifi model.
%   2. Altitude and velocity are hardcoded. This is to make is possible to
%      iteretate the script without human input.
%   3. Not displaying output for the same reason as change 2.

clearvars -except x_a time_vector x_a_options styles idx altitude velocity simulink_block;

addpath obsmutoolsfornewermatlabversions -END % required for some new MATLAB versions

global fi_flag_Simulink

newline = sprintf('\n');

% Remove '_trim' to linearize the original model without the accelerometer
% output.  

%% Trim aircraft to desired altitude and velocity
%%
% altitude = 15000; % input('Enter the altitude for the simulation (ft)  :  ');
% velocity = 500; % input('Enter the velocity for the simulation (ft/s):  ');

%% Initial guess for trim
%%
thrust = 5000;          % thrust, lbs
elevator = -0.09;       % elevator, degrees
alpha = 8.49;              % AOA, degrees
rudder = -0.01;             % rudder angle, degrees
aileron = 0.01;            % aileron, degrees

%% Find trim for Lofi model at desired altitude and velocity
%%
disp('Trimming Low Fidelity Model:');
fi_flag_Simulink = 0;
[trim_state_lo, trim_thrust_lo, trim_control_lo, dLEF, xu_lo] = trim_F16_5(thrust, elevator, alpha, aileron, rudder, velocity, altitude);

%% Find the state space model for the lofi model at the desired alt and vel.
%%
trim_state_lin = trim_state_lo; trim_thrust_lin = trim_thrust_lo; trim_control_lin = trim_control_lo;
[A_lo,B_lo,C_lo,D_lo] = linmod(simulink_block, [trim_state_lin; trim_thrust_lin; trim_control_lin(1); trim_control_lin(2); trim_control_lin(3);...
		dLEF; -trim_state_lin(8)*180/pi], [trim_thrust_lin; trim_control_lin(1); trim_control_lin(2); trim_control_lin(3)]);

%% Make state space model
%%
SS_lo = ss(A_lo,B_lo,C_lo,D_lo);


%% Make MATLAB matrix
%% 
mat_lo = [A_lo B_lo; C_lo D_lo];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Longitudal Directional %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Select the components that make up the longitude A matrix
%%
A_longitude_lo = mat_lo([3 5 7 8 11 13 14], [3 5 7 8 11 13 14]);

%% Select the components that make up the longitude B matrix
%%
B_longitude_lo = mat_lo([3 5 7 8 11 13 14], [19 20]);

%% Select the components that make up the longitude C matrix
%%
C_longitude_lo = mat_lo([21 23 25 26 29], [3 5 7 8 11 13 14]);

%% Select the components that make up the longitude D matrix
%%
D_longitude_lo = mat_lo([21 23 25 26 29], [19 20]);

SS_long_lo = ss(A_longitude_lo, B_longitude_lo, C_longitude_lo, D_longitude_lo);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Lateral Directional %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Select the components that make up the lateral A matrix
%%
A_lateral_lo = mat_lo([4 6 7 9 10 12 13 15 16], [4 6 7 9 10 12 13 15 16]);

%% Select the components that make up the lateral B matrix
%%
B_lateral_lo = mat_lo([4 6 7 9 10 12 13 15 16], [19 21 22]);

%% Select the components that make up the lateral C matrix
%%
C_lateral_lo = mat_lo([22 24 25 27 28 30], [4 6 7 9 10 12 13 15 16]);

%% Select the components that make up the lateral D matrix
%%
D_lateral_lo = mat_lo([22 24 25 27 28 30], [19 21 22]);

SS_lat_lo = ss(A_lateral_lo, B_lateral_lo, C_lateral_lo, D_lateral_lo);

%% Make longitudal direction SYSTEM matrix
%%
sys_long_lo = pck(A_longitude_lo, B_longitude_lo, C_longitude_lo, D_longitude_lo);

%% Make lateral direction SYSTEM matrix and Find poles for lofi
%%
sys_lat_lo = pck(A_lateral_lo, B_lateral_lo, C_lateral_lo, D_lateral_lo);

long_poles_lo = spoles(sys_long_lo);
lat_poles_lo = spoles(sys_lat_lo);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display results

% Don't display
return

clc;

disp(sprintf('Altitude: %.3f ft.', altitude));
disp(sprintf('Velocity: %.3f ft/s\n\n', velocity));

disp('For LOFI Model:  ');
disp('Longitudal Direction:  ');
disp(newline);

disp('A =')
for i=1:length( A_longitude_lo(:,1) )
    mprintf([ A_longitude_lo(i,:) ],'  %.3e ')
end %for

disp('B =')
for i=1:length( B_longitude_lo(:,1) )
    mprintf([ B_longitude_lo(i,:) ],'  %.3e ')
end %for

disp('C =')
for i=1:length( C_longitude_lo(:,1) )
    mprintf([ C_longitude_lo(i,:) ],'  %.3e ')
end %for

disp('D =')
for i=1:length( D_longitude_lo(:,1) )
    mprintf([ D_longitude_lo(i,:) ],'  %.3e ')
end %for

% Display the real, imaginary, frequency (magnitude) and damping ratios
rifd(long_poles_lo)

disp(newline);

disp('Lateral Direaction:  ');

disp(newline);

disp('A =')
for i=1:length( A_lateral_lo(:,1) )
    mprintf([ A_lateral_lo(i,:) ],'  %.3e ')
end %for

disp('B =')
for i=1:length( B_lateral_lo(:,1) )
    mprintf([ B_lateral_lo(i,:) ],'  %.3e ')
end %for

disp('C =')
for i=1:length( C_lateral_lo(:,1) )
    mprintf([ C_lateral_lo(i,:) ],'  %.3e ')
end %for

disp('D =')
for i=1:length( D_lateral_lo(:,1) )
    mprintf([ D_lateral_lo(i,:) ],'  %.3e ')
end %for

% Display the real, imaginary, frequency (magnitude) and damping ratios
rifd(lat_poles_lo)

%% All Poles
figure(1); 
pzmap(SS_lo, 'b');
title_string = sprintf('Altitude = %.2f ft Velocity = %.2f ft/s\nAll Poles\n Blue = lofi', altitude, velocity);
title(title_string);
sgrid;

%% Long. Poles
%%
figure(2); 
pzmap(SS_long_lo, 'b');
title_string = sprintf('Altitude = %.2f ft Velocity = %.2f ft/s\nLongitudal Directional Poles\n Blue = lofi', altitude, velocity);
title(title_string);
sgrid;

%% Lat. Poles
%%
figure(3); 
pzmap(SS_lat_lo, 'b');
title_string = sprintf('Altitude = %.2f ft Velocity = %.2f ft/s\nLateral Directional Poles\n Blue = lofi', altitude, velocity);
title(title_string);
sgrid;

% Create Bode Plots

omega = logspace(-2,2,100);

sysg_lat_lo = frsp(sys_lat_lo,omega);

sysg_long_lo = frsp(sys_long_lo,omega);

figure;
BodeCount = 0;
for state = 1:1:5
    for control = 1:1:2
        BodeCount = BodeCount +1;
        title_string = sprintf('Bode Plot #%d\n State = %d\n Control = %d', BodeCount,state,control);
        vplot('bode', sel(sysg_long_lo,state,control), 'r');
        disp(title_string);
        legend('lofi');
        pause;
    end
end

for state = 1:1:6
    for control = 1:1:3
        BodeCount = BodeCount + 1;
        title_string = sprintf('Bode Plot #%d\n State = %d\n Control = %d', BodeCount,state,control);
        vplot('bode', sel(sysg_lat_lo,state,control), 'r');
        disp(title_string);
        legend('lofi');
        pause;
    end
end
