o_px = [2021 1778 1];

K = [3275 0 2016; 0 3275 1512; 0 0 1];
K_inv = inv(K);
T_inv = o_px * K_inv;
T_inv = T_inv .* 111

T = [68.5; 60.3; 0];
R = [-0.6356 0.77 0.04; -0.54 -0.49 0.69; 0.55 0.42 0.72];

K * R * T;


R = [cos(t) 0 sin(t); 0 1 0; -sin(t) 0 cos(t)];
world = [x; y; z];

R * world;