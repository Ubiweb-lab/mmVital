function y = tqwt_bands(b,w,Q,r,N)
% y = tqwt_bands(b,w,Q,r,N)
% Reconstruction from subsets of TQWT subbands
% using tqwt_radix2
% INPUT
%   b - cell array (length K) of subbands
%   w - TQWT coefficients
%   Q, r - TQWT parameters
%   N - length of signal.
% OUTPUT
%   y - array of reconstructed signals (one per row)
%
% % Example
% x = test_signal(2);
% N = length(x);
% Q = 3; r = 3; J = 18;
% w = tqwt_radix2(x, Q, r, J);
% b = {7:9, 11:13, 15:17};  % tqwt subbands from which to reconstruct
% y = tqwt_bands(b, w, Q, r, N);
% figure(1), clf
% plot(1:N, y(1,:), 1:N, y(2,:)-1, 1:N, y(3,:)-2)
% title('Reconstructed signals'), xlim([0 N]), box off

K = length(b);

y = zeros(K,N);

J = length(w) - 1;

wz = cell(1,J+1);
for j = 1:J+1
    wz{j} = zeros(size(w{j}));
end

for k = 1:K
    bk = b{k};
    for i = 1:length(bk)
        m = bk(i);
        w2 = wz;
        w2{m} = w{m};
        y(k,:) = y(k,:) + itqwt_radix2(w2,Q,r,N);
    end
end



