function [rangeAzimuth, idx] = getOccupByteHeatMap(bytevec, idx, numRangeBins, numAngleBins) %==>>
len = numRangeBins * numAngleBins;
rangeAzimuth = bytevec(idx+1:idx+len);
idx = idx + len;

% group 2 bytes typecase to single
rangeAzimuth = typecast(uint8(rangeAzimuth), 'uint8');
return