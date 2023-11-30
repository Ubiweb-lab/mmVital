function [rangeAzimuth, idx] = getOccupShortHeatMap(bytevec, idx, numRangeBins, numAngleBins) %==>>
len = numRangeBins * numAngleBins * 2;
rangeAzimuth = bytevec(idx+1:idx+len);
idx = idx + len;

% group 2 bytes typecase to single
rangeAzimuth = typecast(uint8(rangeAzimuth), 'uint16');
return