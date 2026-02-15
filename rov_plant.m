function dx = rov_plant(state, thrust, params)
% 6-DOF ROV 动力学（Fossen 形式）

eta = state(1:6);
nu  = state(7:12);

phi = eta(4); theta = eta(5); psi = eta(6);
u = nu(1); v = nu(2); w = nu(3);
p = nu(4); q = nu(5); r = nu(6);

% 质量矩阵
M = diag([
    params.m - params.Xu_dot;
    params.m - params.Yv_dot;
    params.m - params.Zw_dot;
    params.Ixx - params.Kp_dot;
    params.Iyy - params.Mq_dot;
    params.Izz - params.Nr_dot
]);

% 科氏矩阵
a1 = params.m - params.Xu_dot;
a2 = params.m - params.Yv_dot;
a3 = params.m - params.Zw_dot;
C = zeros(6,6);
C(1,5) = a3 * w;
C(1,6) = -a2 * v;
C(2,4) = -a3 * w;
C(2,6) = a1 * u;
C(3,4) = a2 * v;
C(3,5) = -a1 * u;
C = C - C';

% 阻尼矩阵
D = -diag([
    params.Xu + params.Xuu*abs(u);
    params.Yv + params.Yvv*abs(v);
    params.Zw + params.Zww*abs(w);
    params.Kp + params.Kpp*abs(p);
    params.Mq + params.Mqq*abs(q);
    params.Nr + params.Nrr*abs(r)
]);

% 重力和浮力
g = controller_pid_gravity(phi, theta, params);

tau = params.T * thrust;
nu_dot = M \ (tau - C*nu - D*nu - g);

J = [
    cos(psi)*cos(theta), -sin(psi)*cos(phi)+cos(psi)*sin(theta)*sin(phi), sin(psi)*sin(phi)+cos(psi)*cos(phi)*sin(theta);
    sin(psi)*cos(theta), cos(psi)*cos(phi)+sin(phi)*sin(theta)*sin(psi), -cos(psi)*sin(phi)+sin(theta)*sin(psi)*cos(phi);
    -sin(theta),          cos(theta)*sin(phi),                             cos(theta)*cos(phi)
];
R = [
    1, sin(phi)*tan(theta), cos(phi)*tan(theta);
    0, cos(phi),            -sin(phi);
    0, sin(phi)/cos(theta), cos(phi)/cos(theta)
];
eta_dot = [J*nu(1:3); R*nu(4:6)];

dx = [eta_dot; nu_dot];
end

function g = controller_pid_gravity(phi, theta, params)
% 与 controller_pid 中保持一致
W = params.W;
B = params.B;
rG = params.rG;
rB = params.rB;

sph = sin(phi); cph = cos(phi);
sth = sin(theta); cth = cos(theta);

g = zeros(6,1);
g(1) = (W - B) * sth;
g(2) = -(W - B) * cth * sph;
g(3) = -(W - B) * cth * cph;
g(4) = (rG(2)*W - rB(2)*B)*cth*cph - (rG(3)*W - rB(3)*B)*cth*sph;
g(5) = -(rG(3)*W - rB(3)*B)*sth - (rG(1)*W - rB(1)*B)*cth*cph;
g(6) = (rG(1)*W - rB(1)*B)*cth*sph + (rG(2)*W - rB(2)*B)*sth;
end

