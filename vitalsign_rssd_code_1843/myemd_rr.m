function [y1, y2,w1,Q1] = myemd_rr(x)
N = length(x);

%% RSSD parameter initialization
    Q1 = 4;
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

[~,~,w1] = dualQd(x,Q1,r1,J1,Q2,r2,J2,lam1,lam2,mu,Nit);

                e=PlotEnergy(w1); %energy distribution in levels
                fs=10;
                fc=tqwt_fc(Q1,r1,J1,fs); %centerfrequency of levels
                rr_test=zeros(2,J1);
                rr_test(1,:)=fc;
                rr_test(2,:)=e(1,1:J1);
                rr_test_enrgy=rr_test(:,(rr_test(1,:)>=0.1));
                rr_test_enrgy_1=rr_test_enrgy(:,(rr_test_enrgy(1,:)<0.25));
                enrgy_1=sum(rr_test_enrgy_1(2,:));
                rr_test_enrgy_2=rr_test_enrgy(:,(rr_test_enrgy(1,:)<0.33));
                enrgy_2=sum(rr_test_enrgy_2(2,:))-enrgy_1;
                rr_test_enrgy_3=rr_test_enrgy(:,(rr_test_enrgy(1,:)<0.42));
                enrgy_3=sum(rr_test_enrgy_3(2,:))-enrgy_2-enrgy_1;
                rr_test_enrgy_4=rr_test_enrgy(:,(rr_test_enrgy(1,:)<0.6));
                enrgy_4=sum(rr_test_enrgy_4(2,:))-enrgy_3-enrgy_2-enrgy_1;

                %energy distribution in different frequency HR ranges
                %according to which final RSSD decomposition shall be done
                rr_energy(1,1)=enrgy_1*100/(enrgy_1+enrgy_2+enrgy_3+enrgy_4);
                rr_energy(1,2)=enrgy_2*100/(enrgy_1+enrgy_2+enrgy_3+enrgy_4);
                rr_energy(1,3)=enrgy_3*100/(enrgy_1+enrgy_2+enrgy_3+enrgy_4);
                rr_energy(1,4)=enrgy_4*100/(enrgy_1+enrgy_2+enrgy_3+enrgy_4);
                rr_energy=round(rr_energy);
                q=load('Qfile_RR.mat');
                q_test=q.q_RR_all;
                q_test1=q_test(:,1:4);
                for m=1:324
                    d(m)=norm(q_test1(m,1:4)-rr_energy);
                end
                [~,indB]=min(d);
                Q1=q_test(indB,5);

 [y1,y2,w1] = dualQd(x,Q1,r1,J1,Q2,r2,J2,lam1,lam2,mu,Nit);

end