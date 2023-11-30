function [tlv, idx] = getTlv(bytevec, idx)
word = [1 256 65536 16777216]';
tlv.type = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
tlv.length = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
return