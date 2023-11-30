function [sphandle] = configureSport(comportSnum)

if ~isempty(instrfind('Type','serial'))
    disp('Serial port(s) already open. Re-initializing...');
    delete(instrfind('Type','serial'));  % delete open serial ports.
end
comportnum_str=['COM' num2str(comportSnum)]
sphandle = serial(comportnum_str,'BaudRate',921600);
set(sphandle,'InputBufferSize', 2^16);
set(sphandle,'Timeout',10);
set(sphandle,'ErrorFcn',@dispError);
set(sphandle,'BytesAvailableFcnMode','byte');
set(sphandle,'BytesAvailableFcnCount', 2^16+1);%BYTES_AVAILABLE_FCN_CNT);
set(sphandle,'BytesAvailableFcn',@readUartCallbackFcn);
fopen(sphandle);

return