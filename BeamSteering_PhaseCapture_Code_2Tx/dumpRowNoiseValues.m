function [idx] = dumpRowNoiseValues(bytevec, idx, numRangeBins) %==>>
global rowInit;
%    len = 6; % uint8_t
%    idx = idx + len;

rowInit = 2;
len = numRangeBins * 4; % 1 single per row
rowNoise = bytevec(idx+1:idx+len);
idx = idx + len;

% group 4 bytes typecase to single
rowNoise = typecast(uint8(rowNoise), 'single');
fprintf("\n");
row = 1;

for grp = 1:numRangeBins / 8
    fprintf("rowNoise %2d %d ", (grp-1)*8, 8);
    for ridx = 1:8
        fprintf(" %f", rowNoise(row));
        row = row + 1;
    end

    fprintf("\n");
end

return
