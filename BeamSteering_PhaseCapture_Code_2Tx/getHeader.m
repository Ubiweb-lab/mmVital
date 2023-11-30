function [Header, idx] = getHeader(bytevec, idx, platformType)
idx = idx + 8; %Skip magic word
word = [1 256 65536 16777216]';
Header.totalPacketLen = sum(bytevec(idx+[1:4]) .* word);
idx = idx + 4;
Header.platform = sum(bytevec(idx+[1:4]) .* word);
idx = idx + 4;
Header.frameNumber = sum(bytevec(idx+[1:4]) .* word);
idx = idx + 4;
Header.timeCpuCycles = sum(bytevec(idx+[1:4]) .* word);
idx = idx + 4;
Header.numDetectedObj = sum(bytevec(idx+[1:4]) .* word);
idx = idx + 4;
Header.numTLVs = sum(bytevec(idx+[1:4]) .* word);
idx = idx + 4;
return
