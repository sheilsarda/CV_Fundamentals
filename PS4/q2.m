%% Q2

R_1 = [1 0 0 ; 0 sqrt(3)/2 -1/2; 0 1/2 sqrt(3)/2];
R_2 = [1 0 0 ; 0 1/2 -sqrt(3)/2; 0 sqrt(3)/2 1/2];

H =[-4/8 0 1; 
    6/8 (sqrt(3) - 12)/8 (5 - 4*sqrt(3))/8;
    -2*sqrt(3)/8 (7+4*sqrt(3))/8 (sqrt(3) + 4)/8];

Hprime = inv(R_2) * H * R_1;

theta = 2*pi/3;
R_y = [cos(2*pi/3) 0 sin(2*pi/3); 0 1 0; -sin(2*pi/3) 0 cos(2*pi/3)];

R12 = R_1 * R_y * inv(R_2);
Tprime = [1; 0 ; 4];

T = R_1 * Tprime;

R12 + T/2;

%% Q1

A = [-1.25 -0.25 -0.25 1.75; -0.5 -1.5 -0.5 2.5];
B = [-0.65 -1.52 -0.53 2.7; 1.175 0.065 0.185 -1.43];
mat = A * transpose(B);
[U S V] = svd(mat);
Vt = transpose(V);
R  =Vt * [1 0; 0 det(Vt * transpose(U))] * transpose(U)