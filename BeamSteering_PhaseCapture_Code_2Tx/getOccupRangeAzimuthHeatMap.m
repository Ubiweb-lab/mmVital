function [rangeAzimuth, idx] = getOccupRangeAzimuthHeatMap(bytevec, idx, numRangeBins, numAngleBins) %==>>
len = numRangeBins * numAngleBins * 4;
rangeAzimuth = bytevec(idx+1:idx+len);
idx = idx + len;

% group 4 bytes typecase to single
rangeAzimuth = typecast(uint8(rangeAzimuth), 'single');
return
