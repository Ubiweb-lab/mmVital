function [y] = pow2roundup (x)
y = 1;
while x > y
    y = y * 2;
end
return