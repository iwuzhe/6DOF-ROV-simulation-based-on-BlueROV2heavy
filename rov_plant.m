function dx = rov_plant(state, thrust, tau_env, params)
% 6-DOF ROV 动力学

eta = state(1:6);
nu  = state(7:12);
phi = eta(4); theta = eta(5); psi = eta(6);
u = nu(1); v = nu(2); w = nu(3);
p = nu(4); q = nu(5); r = nu(6);

% [标准 Fossen Notation kept for Physics]
M = diag([
    params.m - params.Xu_dot; params.m - params.Yv_dot; params.m - params.Zw_dot;
    params.Ixx - params.Kp_dot; params.Iyy - params.Mq_dot; params.Izz - params.Nr_dot
]);

% 科里奥利力
a1 = params.m - params.Xu_dot;
a2 = params.m - params.Yv_dot;
a3 = params.m - params.Zw_dot;
C = zeros(6,6);
C(1,5) = a3 * w; C(1,6) = -a2 * v;
C(2,4) = -a3 * w; C(2,6) = a1 * u;
C(3,4) = a2 * v; C(3,5) = -a1 * u;
C = C - C';

% 阻尼 
D = -diag([
    params.Xu + params.Xuu*abs(u);
    params.Yv + params.Yvv*abs(v);
    params.Zw + params.Zww*abs(w);
    params.Kp + params.Kpp*abs(p);
    params.Mq + params.Mqq*abs(q);
    params.Nr + params.Nrr*abs(r)
]);

% 重力恢复力矩
g = zeros(6,1);
W = params.W; B = params.B; rG = params.rG; rB = params.rB;
sph=sin(phi); cph=cos(phi); sth=sin(theta); cth=cos(theta);
g(1)=(W-B)*sth; g(2)=-(W-B)*cth*sph; g(3)=-(W-B)*cth*cph;
g(4)=(rG(2)*W-rB(2)*B)*cth*cph - (rG(3)*W-rB(3)*B)*cth*sph;
g(5)=-(rG(3)*W-rB(3)*B)*sth - (rG(1)*W-rB(1)*B)*cth*cph;
g(6)=(rG(1)*W-rB(1)*B)*cth*sph + (rG(2)*W-rB(2)*B)*sth;

% 推力映射 (使用 prop_T_matrix)
tau_thruster = params.prop_T_matrix * thrust;
tau = tau_thruster + tau_env;

% 动力学方程
nu_dot = M \ (tau - C*nu - D*nu - g);

% 运动学方程
J = [cos(psi)*cos(theta), -sin(psi)*cos(phi)+cos(psi)*sin(theta)*sin(phi), sin(psi)*sin(phi)+cos(psi)*cos(phi)*sin(theta);
     sin(psi)*cos(theta), cos(psi)*cos(phi)+sin(phi)*sin(theta)*sin(psi), -cos(psi)*sin(phi)+sin(theta)*sin(psi)*cos(phi);
     -sin(theta),          cos(theta)*sin(phi),                             cos(theta)*cos(phi)];
R_ang = [1, sin(phi)*tan(theta), cos(phi)*tan(theta);
         0, cos(phi),            -sin(phi);
         0, sin(phi)/cos(theta), cos(phi)/cos(theta)];
     
eta_dot = [J*nu(1:3); R_ang*nu(4:6)];

dx = [eta_dot; nu_dot];
end