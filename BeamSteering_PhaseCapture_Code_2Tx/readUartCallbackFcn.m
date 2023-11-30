function [] = readUartCallbackFcn(obj, event)
global bytevecAcc;
global bytevecAccLen;
global readUartFcnCntr;
global BYTES_AVAILABLE_FLAG

global BYTE_VEC_ACC_MAX_SIZE

bytesToRead = get(obj,'BytesAvailable');
if(bytesToRead == 0)
    return;
end

[bytevec, byteCount] = fread(obj, bytesToRead, 'uint8');

if bytevecAccLen + length(bytevec) < BYTE_VEC_ACC_MAX_SIZE * 3/4
    bytevecAcc(bytevecAccLen+1:bytevecAccLen+byteCount) = bytevec;
    bytevecAccLen = bytevecAccLen + byteCount;
else
    bytevecAccLen = 0;
end

readUartFcnCntr = readUartFcnCntr + 1;
BYTES_AVAILABLE_FLAG = 1;

return