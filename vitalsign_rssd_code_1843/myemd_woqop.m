function [y1, y2,w1,Q1] = myemd_woqop(x)
N = length(x);

%% RSSD parameter initialization
%     Q1 = 10.88;
    Q1 = 8.523809524;
    r1 = 3;
    J1 = 27;
        Q2 = 1.105263158;
        r2 = 3;
        J2 = 8;
% Set MCA parameters
        Nit = 100;          % Number of iterations
        mu = 0.5;           % SALSA parameter
        theta1=0.5;       %normalization parameter
        theta2=0.5;

%% Peform decomposition

now1 = ComputeNow(N,Q1,r1,J1,'radix2');
now2 = ComputeNow(N,Q2,r2,J2,'radix2');
lam1=theta1*now1;
lam2=theta2*now2;

[y1,y2,w1] = dualQd(x,Q1,r1,J1,Q2,r2,J2,lam1,lam2,mu,Nit);

end