function [y1, y2,w1,Q1] = myemd(x)
N = length(x);

%% RSSD parameter initialization
    Q1 = 10.88;
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
                hr_test=zeros(2,J1);
                hr_test(1,:)=fc;
                hr_test(2,:)=e(1,1:J1);
                hr_test_enrgy=hr_test(:,(hr_test(1,:)>=1));
                hr_test_enrgy_1=hr_test_enrgy(:,(hr_test_enrgy(1,:)<1.25));
                enrgy_1=sum(hr_test_enrgy_1(2,:));
                hr_test_enrgy_2=hr_test_enrgy(:,(hr_test_enrgy(1,:)<1.5));
                enrgy_2=sum(hr_test_enrgy_2(2,:))-enrgy_1;
                hr_test_enrgy_3=hr_test_enrgy(:,(hr_test_enrgy(1,:)<1.75));
                enrgy_3=sum(hr_test_enrgy_3(2,:))-enrgy_2-enrgy_1;
                hr_test_enrgy_4=hr_test_enrgy(:,(hr_test_enrgy(1,:)<2));
                enrgy_4=sum(hr_test_enrgy_4(2,:))-enrgy_3-enrgy_2-enrgy_1;

                %energy distribution in different frequency HR ranges
                %according to which final RSSD decomposition shall be done
                hr_energy(1,1)=enrgy_1*100/(enrgy_1+enrgy_2+enrgy_3+enrgy_4);
                hr_energy(1,2)=enrgy_2*100/(enrgy_1+enrgy_2+enrgy_3+enrgy_4);
                hr_energy(1,3)=enrgy_3*100/(enrgy_1+enrgy_2+enrgy_3+enrgy_4);
                hr_energy(1,4)=enrgy_4*100/(enrgy_1+enrgy_2+enrgy_3+enrgy_4);
                hr_energy=round(hr_energy);
                q=load('Qfile.mat');
                q_test=q.q_heart_all;
                q_test1=q_test(:,1:4);
                for m=1:324
                    d(m)=norm(q_test1(m,1:4)-hr_energy);
                end
                [~,indB]=min(d);
                Q1=q_test(indB,5);
                if Q1<5
                    J1=22;
                end

 [y1,y2,w1] = dualQd(x,Q1,r1,J1,Q2,r2,J2,lam1,lam2,mu,Nit);

end