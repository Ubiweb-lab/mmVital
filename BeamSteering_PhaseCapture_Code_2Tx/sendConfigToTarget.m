function sendConfigToTarget(comportCliNum, cliCfg, cliCfgFileName)
spCliHandle = configureCliPort(comportCliNum);

warning off; %MATLAB:serial:fread:unsuccessfulRead
timeOut = get(spCliHandle,'Timeout');
set(spCliHandle,'Timeout',1);
% tStart = tic;
while 1
    fprintf(spCliHandle, '');
    cc = fread(spCliHandle,100);
    cc = strrep(strrep(cc,char(10),''),char(13),'');
    if ~isempty(cc)
        break;
    end
    pause(0.1);
    % toc(tStart);
end
set(spCliHandle,'Timeout', timeOut);
warning on;

% Send CLI configuration to XWR1xxx
fprintf('Sending configuration to XWR1xxx %s ...\n', cliCfgFileName);
for k = 1:length(cliCfg)
    if isempty(strrep(strrep(cliCfg{k},char(9),''),char(32),''))
        continue;
    end
    if strcmp(cliCfg{k}(1),'%')
        continue;
    end
    fprintf(spCliHandle, cliCfg{k});
    fprintf('%s\n', cliCfg{k});
    for kk = 1:3
        cc = fgetl(spCliHandle);
        if strcmp(cc,'Done')
            fprintf('%s\n',cc);
            break;
        elseif ~isempty(strfind(cc, 'not recognized as a CLI command'))
            fprintf('%s\n',cc);
            return;
        elseif ~isempty(strfind(cc, 'Error'))
            fprintf('%s\n',cc);
            return;
        end
    end
    pause(0.2)
end
fclose(spCliHandle);
delete(spCliHandle);

return