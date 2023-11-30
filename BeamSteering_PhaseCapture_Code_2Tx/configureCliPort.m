function [sphandle] = configureCliPort(comportPnum)
%if ~isempty(instrfind('Type','serial'))
%    disp('Serial port(s) already open. Re-initializing...');
%    delete(instrfind('Type','serial'));  % delete open serial ports.
%end
comportnum_str = ['COM' num2str(comportPnum)]
sphandle = serial(comportnum_str, 'BaudRate', 115200);
set(sphandle, 'Parity', 'none');
set(sphandle, 'Terminator', 'LF');

fopen(sphandle);

return