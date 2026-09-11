%% Magnetic Levitation (MagLev) Simulation & LQR Control
clear; clc; close all;

% 1. System Parameters & Operating Point
p.m = 0.05;       % Ball mass (kg)
p.g = 9.81;       % Gravity (m/s^2)
p.K = 1e-4;       % Force constant (N*m^2 / A^2)
p.R = 10.0;       % Coil resistance (Ohms)
p.L = 0.1;        % Coil inductance (Henry)
p.V_max = 24.0;   % Max voltage limit (V)
p.V_min = 0.0;    % Min voltage limit (V)

p.x0 = 0.01;                        % Equilibrium position (1 cm)
p.i0 = p.x0 * sqrt((p.m * p.g) / p.K); % Nominal current (A)
p.v0 = p.R * p.i0;                  % Nominal voltage (V)

% 2. Linearization & LQR Gain Calculation
Cx = (2 * p.K * p.i0^2) / (p.m * p.x0^3);
Ci = (2 * p.K * p.i0) / (p.m * p.x0^2);

A = [ 0,     1,       0;
     Cx,     0,     -Ci;
      0,     0, -p.R/p.L];
B = [0; 0; 1/p.L];

% Augmented state space for integral action
A_aug = [A, [0; 0; 0]; 1, 0, 0, 0];
B_aug = [B; 0];
Q = diag([1e5, 10, 1, 1e7]);
R_weight = 1;

K_lqr = lqr(A_aug, B_aug, Q, R_weight);

% 3. Simulation Setup
tspan = [0, 2];               % Simulation time: 2 seconds
x_init = [0.015; 0; 0; 0];    % Start at 1.5 cm distance

% Step profile reference target (cm converted to m)
x_ref = @(t) 0.01 + 0.003 * (t >= 0.8) - 0.005 * (t >= 1.4);

% 4. Run Non-Linear Integration
[t, state] = ode45(@(t, s) maglev_dynamics(t, s, x_ref(t), K_lqr, p), tspan, x_init);

% 5. Process Output Signals
N = length(t);
v_control = zeros(N, 1);
ref_signal = zeros(N, 1);

for k = 1:N
    ref = x_ref(t(k));
    ref_signal(k) = ref;

    dx   = state(k, 1) - ref;
    dvel = state(k, 2);
    di   = state(k, 3) - p.i0;
    e_i  = state(k, 4);

    u_delta = -K_lqr * [dx; dvel; di; e_i];
    v_act   = p.v0 + u_delta;
    v_control(k) = min(max(v_act, p.V_min), p.V_max);
end

% 6. Plot Performance Graphs
figure('Name', 'MagLev Simulation', 'Color', [1 1 1]);

subplot(3, 1, 1);
plot(t, state(:, 1) * 100, 'b', 'LineWidth', 1.8); hold on;
plot(t, ref_signal * 100, 'r--', 'LineWidth', 1.5);
ylabel('Position (cm)');
title('Magnetic Levitation - Position Tracking');
legend('Actual Position', 'Target Reference', 'Location', 'best');
grid on;

subplot(3, 1, 2);
plot(t, state(:, 3), 'm', 'LineWidth', 1.5);
ylabel('Current (A)');
title('Electromagnet Current');
grid on;

subplot(3, 1, 3);
plot(t, v_control, 'k', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Voltage (V)');
title('Control Input Voltage (Saturated [0V - 24V])');
grid on;

%% --- Local Dynamics Function ---
function ds = maglev_dynamics(~, s, r, K_lqr, p)
    x   = s(1); % Ball position (m)
    v_x = s(2); % Ball velocity (m/s)
    i   = s(3); % Current (A)
    e_i = s(4); % Integrated error

    % Error states
    dx   = x - r;
    dvel = v_x;
    di   = i - p.i0;

    % Saturation control logic
    u_delta = -K_lqr * [dx; dvel; di; e_i];
    v = min(max(p.v0 + u_delta, p.V_min), p.V_max);

    x = max(x, 1e-4); % Prevent non-physical boundary overlap

    % Non-linear differential equations
    d_x  = v_x;
    d_vx = p.g - (p.K / p.m) * (i / x)^2;
    d_i  = (v - p.R * i) / p.L;
    d_ei = dx;

    ds = [d_x; d_vx; d_i; d_ei];
end