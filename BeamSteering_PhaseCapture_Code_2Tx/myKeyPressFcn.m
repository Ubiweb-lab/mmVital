function myKeyPressFcn(hObject, event)
global EXIT_KEY_PRESSED
if lower(event.Key) == 'q'
    EXIT_KEY_PRESSED  = 1;
    cliCfgFileName="1843_Sensor_Stop.cfg";
    cliCfg = readConfigFile(cliCfgFileName);
    %% CLI comport number = 4
    sendConfigToTarget(4, cliCfg, cliCfgFileName);
end

return
