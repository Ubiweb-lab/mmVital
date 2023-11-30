%% VitalSign_PhaseData_Capture_1843('9','8','VitalSign_1843_withoutSteering.cfg','2')
%% 9 is data port and 8 is cli port

function [] = VitalSign_PhaseData_Capture_1843(comportSnum, comportCliNum, cliCfgFileName, polarPlotMode)

%% %%% Initialization %%%%%%%%%%%%%%
if nargin<4, polarPlotMode = '0'; end

if(ischar(comportSnum))
    comportSnum         = str2num(comportSnum);
    polarPlotMode       = str2num(polarPlotMode);
end
loadCfg     = 1;
debugFlag   = 0;

file_path = "C:\Users\Anuradha\Desktop\RA_Maori Vital Sign Study\Code\Data\D2\";
filename_time = strcat(file_path, "Time_Log.txt");
filename_phase = strcat(file_path, "Phase_save.xlsx");

%% Global parameters
global platformType MAX_NUM_OBJECTS bytevecAcc OBJ_STRUCT_SIZE_BYTES STATS_SIZE_BYTES BYTES_AVAILABLE_FCN_CNT EXIT_KEY_PRESSED
global BYTE_VEC_ACC_MAX_SIZE bytevecAccLen readUartFcnCntr BYTES_AVAILABLE_FLAG activeFrameCPULoad
global interFrameCPULoad guiCPULoad guiProcTime matFileObj %==>>
global Params zonePwr zonePwrdB figure_width figure_height rollingMax rollingAvg rollingIdx rowInit



%%%%----- %%%%
rollingMax = [500 500 500];
rollingAvg = 500;
rollingIdx = 1;
rowInit = 0;detObj=[];
platformType = hex2dec('a1843');
MAX_NUM_OBJECTS         = 100;
OBJ_STRUCT_SIZE_BYTES   = 12;
STATS_SIZE_BYTES        = 16;
BYTES_AVAILABLE_FCN_CNT = 32*8;
EXIT_KEY_PRESSED = 0;
BYTE_VEC_ACC_MAX_SIZE = 2^16; 
bytevecAcc      = zeros(BYTE_VEC_ACC_MAX_SIZE,1);
bytevecAccLen   = 0;
readUartFcnCntr = 0;
BYTES_AVAILABLE_FLAG = 0;

activeFrameCPULoad = zeros(100,1);
interFrameCPULoad = zeros(100,1);
guiCPULoad = zeros(100,1);
guiProcTime = 0;
matFileObj = [];   
% PLOT_DISPLAY_LENGTH = 128;
% MSG_DETECTED_POINTS = 1;
% MSG_RANGE_PROFILE   = 2;
% MSG_NOISE_PROFILE   = 3;
% MSG_AZIMUT_STATIC_HEAT_MAP = 4;
% MSG_RANGE_DOPPLER_HEAT_MAP = 5;
MSG_STATS = 6;
MSG_RANGE_AZIMUT_HEAT_MAP = 8;
MSG_OD_DECISION = 9;
VS_OUTPUT_HEART_BREATHING_RATES = 10;
MSG_OD_ROW_NOISE = 11;
NUM_RANGE_BINS_IN_HEATMAP = 64;
NUM_ANGLE_BINS = 48;
% % NUM_MAX_FRAMES_LOG = 192;
bytevec_cp_max_len = 2^16; %==>>
bytevec_cp = zeros(bytevec_cp_max_len,1);
bytevec_cp_len = 0;
displayUpdateCntr = 0;
packetNumberPrev = 0;
decisionValue = [0, 0, 0, 0, 0, 0];
extractedValue = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
occ_color = [1.0, 0.0, 0.0]; % red for AWR1843
timeout_ctr = 0;
time_exp=1;


%% Setup the main figure
figHnd = figure(1);clf(figHnd);
set(figHnd,'Name',' Range-Angle Heat Map','NumberTitle','off')
warning off MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame
jframe = get(figHnd,'javaframe');jIcon = javax.swing.ImageIcon('');jframe.setFigureIcon(jIcon);
set(figHnd, 'MenuBar', 'none');set(figHnd, 'Color', [0.8 0.8 0.8]);
set(figHnd, 'KeyPressFcn', @myKeyPressFcn)
pos = get(gcf, 'Position');
figure_width  = pos(3);figure_height = pos(4);
scrSize = get (0, 'ScreenSize');scrx = scrSize(3);scry = scrSize(4);

%% Read Configuration file
cliCfg = readConfigFile(cliCfgFileName);

%% Parse CLI parameters
Params = parseConfig(cliCfg);
Params.dataPath.numAngleBins = NUM_ANGLE_BINS; %==>>

theta = (-NUM_ANGLE_BINS/2:NUM_ANGLE_BINS/2-1)/NUM_ANGLE_BINS*(60/90)*pi;
theta_degree = (-NUM_ANGLE_BINS/2:NUM_ANGLE_BINS/2-1)/NUM_ANGLE_BINS*(60/90)*180;
range = (0:Params.dataPath.numRangeBins-1) * Params.dataPath.rangeIdxToMeters;

%% Zone structure
for z = 1:Params.numZones
    zone(z) = define_zone(range, theta_degree, Params.zoneDef{z});
end

%Fill a default heatmap with a gradient pattern. If displayed, the target
%code is not sending heatmaps because it is generating noise-floor values.
for row = 1:Params.dataPath.numRangeBins
    for az = 1:NUM_ANGLE_BINS
        rangeAzimuth(row, az) = row * 100.0;
    end
end


%% Configure Data UART port 
sphandle = configureSport(comportSnum);

%% Send Configuration Parameters to AWR1843 (zero Degree)
if loadCfg == 1
    sendConfigToTarget(comportCliNum, cliCfg, cliCfgFileName);
end
%% Plot axis limits set

numZones    = Params.numZones;
winLen      = Params.windowLen;
zonePwr     = zeros(winLen, numZones);  % circular buffer
zonePwrdB   = zeros(winLen, numZones);  % circular buffer
frameIdx    = 0;

%Resize and position the two GUI figures
figx1 = scrx  * 0.33; %heatmap
figy1 = figx1 * 0.60;
figy2 = scry  * 0.60; %plots
figx2 = figy2 * 1.10;
set(figure(1),'Position', [(20+figx2) (scry-figy1)/2 figx1 figy1]); % heatmap


%% User defined variables
global angle_prof i Flag_Beam_Lock phase_cntr
global fin_clt_frm range_prof

i=1;
Flag_Beam_Lock=0;
phase_cntr = 1;
angle_prof=zeros(3,1);
range_prof=zeros(3,1);
ang_est_iter=1;
data_collection_on=1;
target_angle = 0;
Phase_sig = zeros(1512, 1);
target_ra_mat = zeros(2,200);
fin_clt_frm = readmatrix('clutter_save.xlsx');



%% -------------------- Main Loop ------------------------
while (data_collection_on)   
  
    %% beam towards target and lock
    if Flag_Beam_Lock ~=1 && i==2
        target_angle_bin=round(angle_prof);
        target_angle=target_angle_bin*2.5-60;
        fprintf('the target angle bin is=%d\n',target_angle_bin);
        fprintf('the target angle is=%0.2f\n',target_angle);
        target_range_bin=round(range_prof);          
        fprintf('the target range bin is=%d\n',target_range_bin);
        % cliCfg = readConfigFile(cliCfgFileName);
       
        %% zone ang range    
        if target_angle_bin > 3 && target_angle_bin < 43
            zone_angle_bin=target_angle_bin-3;
            var_zone='zoneDef 2 r1 5 a1 7  r1 5 a1 7';
        elseif target_angle_bin >= 43 && target_angle_bin <= 45
            zone_angle_bin=target_angle_bin;
            var_zone='zoneDef 2 r1 5 a1 3  r1 5 a1 3';
        else
            fprintf('Error: You are out of zone\n');
            data_collection_on = 0;
        end
        var_zone_new=strrep(var_zone,'r1',int2str(target_range_bin-2));
        var_zone_new=strrep(var_zone_new,'a1',int2str(zone_angle_bin));
        fprintf('the Zone target angle bin is=%0.2f\n',zone_angle_bin);    
        cliCfg{15}=var_zone_new;
        sendConfigToTarget(comportCliNum, cliCfg, cliCfgFileName);
        Flag_Beam_Lock=1;
          
       
        %% redfine the zones as per target
        Params = parseConfig(cliCfg);
        for z = 1:Params.numZones
            zone(z) = define_zone(range, theta_degree, Params.zoneDef{z});
        end
        
       fid = fopen(filename_time, 'a');
       fprintf(fid, "The starting Time is: %s\n", datetime(now, "ConvertFrom", "datenum"));

    end
    
    %% Read the data
    readUartCallbackFcn(sphandle, 0);

    if BYTES_AVAILABLE_FLAG == 1
        BYTES_AVAILABLE_FLAG = 0;
        %fprintf('bytevec_cp_len, bytevecAccLen = %d %d \n',bytevec_cp_len, bytevecAccLen)
        if (bytevec_cp_len + bytevecAccLen) < bytevec_cp_max_len
            bytevec_cp(bytevec_cp_len+1:bytevec_cp_len + bytevecAccLen) = bytevecAcc(1:bytevecAccLen);
            bytevec_cp_len = bytevec_cp_len + bytevecAccLen;
            bytevecAccLen = 0;
        else
            fprintf('Error: Buffer overflow, bytevec_cp_len, bytevecAccLen = %d %d \n',bytevec_cp_len, bytevecAccLen)
        end
    end

    bytevecStr = char(bytevec_cp(1:bytevec_cp_len));

    magicOk = 0;
    startIdx = strfind(bytevecStr', char([2 1 4 3 6 5 8 7]));
    if ~isempty(startIdx)
        if startIdx(1) > 1
            bytevec_cp(1: bytevec_cp_len-(startIdx(1)-1)) = bytevec_cp(startIdx(1):bytevec_cp_len);
            bytevec_cp_len = bytevec_cp_len - (startIdx(1)-1);
        end
        if bytevec_cp_len < 0
            fprintf('Error: %d %d \n',bytevec_cp_len, bytevecAccLen)
            bytevec_cp_len = 0;
        end

        totalPacketLen = sum(bytevec_cp(8+[1:4]) .* [1 256 65536 16777216]');
        if bytevec_cp_len >= totalPacketLen
            magicOk = 1;
        else
            magicOk = 0;
        end
    end
%% TLV Read and parse the data

    byteVecIdx = 0;
    if(magicOk == 1)
        if debugFlag
            fprintf('Frame Interval = %.3f sec,  ', toc(tStart));
        end
        tStart = tic;

        %% Read the header
        [Header, byteVecIdx] = getHeader(bytevec_cp, byteVecIdx);
        frameIdx = frameIdx + 1;
        detObj.numObj = 0;

        %% Read each TLV
        for tlvIdx = 1:Header.numTLVs

            [tlv, byteVecIdx] = getTlv(bytevec_cp, byteVecIdx);

            switch tlv.type

                case MSG_RANGE_AZIMUT_HEAT_MAP  %==>>
                    switch Params.guiMonitor.rangeAzimuthHeatMap %% should always be 16 (check in config file guimonitor command 3rd value)
                        case 32
                            [rangeAzimuth_vec, byteVecIdx] = getOccupRangeAzimuthHeatMap(bytevec_cp, ...
                                byteVecIdx, ...
                                Params.dataPath.numRangeBins, ...
                                NUM_ANGLE_BINS);  % Params.dataPath.numAngleBins
                        case 16
                            [rangeAzimuth_vec, byteVecIdx] = getOccupShortHeatMap(bytevec_cp, ...
                                byteVecIdx, ...
                                Params.dataPath.numRangeBins, ...
                                NUM_ANGLE_BINS);  % Params.dataPath.numAngleBins
                        case 8
                            [rangeAzimuth_vec, byteVecIdx] = getOccupByteHeatMap(bytevec_cp, ...
                                byteVecIdx, ...
                                Params.dataPath.numRangeBins, ...
                                NUM_ANGLE_BINS);  % Params.dataPath.numAngleBins
                        otherwise
                    end
                    %%====================================================================
                    %% Angle calculation

                    rangeAzimuth = reshape(rangeAzimuth_vec, NUM_ANGLE_BINS, Params.dataPath.numRangeBins).';
                    if ang_est_iter <101 && Flag_Beam_Lock ~=1
                        rangeAzimuth_matrix=double(rangeAzimuth)-double(fin_clt_frm);
                        maximum = max(max(rangeAzimuth_matrix));
                        [range_target, ang_of_arrival]=find(rangeAzimuth_matrix==maximum);
                        target_ra_mat(1,ang_est_iter)=range_target(1,1);
                        target_ra_mat(2,ang_est_iter)=ang_of_arrival(1,1);     
                        fprintf('Range bin = %d, ang_of_arrival = %d\n', range_target, ang_of_arrival);                                              
                        ang_est_iter=ang_est_iter+1;                        
                    elseif ang_est_iter==101 && Flag_Beam_Lock ~=1                      
                        target_ra_mat = target_ra_mat';
                        [ufrq,~,ic] = unique(target_ra_mat(:,1));
                        tally = accumarray(ic, 1);
                        range_bin_temp = [ufrq, tally];
                        
                        range_bin_temp=range_bin_temp(range_bin_temp(:,1)>5,:);
                        range_bin_temp=range_bin_temp(range_bin_temp(:,1)<40,:);
                        [~, raw_ind] = max(range_bin_temp(:, 2));
                        range_prof = range_bin_temp(raw_ind, 1);

                        angle_mat_temp=target_ra_mat(target_ra_mat(:,1) == range_prof,:) ;
                        angle_prof = round(mean(angle_mat_temp(:, 2)));
                                          
                        i=i+1;
                        ang_est_iter=1;
                                              
                    end                   


                case MSG_OD_DECISION  %==>>
                    [decisionValue, byteVecIdx] = getOccupDecision(bytevec_cp, byteVecIdx,numZones);
%                     fprintf("decision value: %d %d\n", decisionValue(1), decisionValue(2));
                    %%====================================================================

                case VS_OUTPUT_HEART_BREATHING_RATES
                    [extractedValue, byteVecIdx] = getVitalSignsHeartBreathingRate(bytevec_cp, byteVecIdx);

                case MSG_OD_ROW_NOISE  %==>>
                    %This message comes only once, and only if rowNoise commands are not send via CLI
                    [byteVecIdx] = dumpRowNoiseValues(bytevec_cp, byteVecIdx, NUM_RANGE_BINS_IN_HEATMAP);

                case MSG_STATS
                    [StatsInfo, byteVecIdx] = getStatsInfo(bytevec_cp, byteVecIdx);
                    % fprintf('StatsInfo: %d, %d, %d %d \n', StatsInfo.interFrameProcessingTime, StatsInfo.transmitOutputTime, StatsInfo.interFrameProcessingMargin, StatsInfo.interChirpProcessingMargin);
                    displayUpdateCntr = displayUpdateCntr + 1;
                    interFrameCPULoad = [interFrameCPULoad(2:end); StatsInfo.interFrameCPULoad];
                    activeFrameCPULoad = [activeFrameCPULoad(2:end); StatsInfo.activeFrameCPULoad];
                    guiCPULoad = [guiCPULoad(2:end); 100*guiProcTime/Params.frameCfg.framePeriodicity];
                    if displayUpdateCntr == 40
                        UpdateDisplayTable(Params);
                        displayUpdateCntr = 0;
                    end

                otherwise

            end

        end % tlvIdx = 1:Header.numTLVs
        
%%  Display create and plot

        if (rowInit == 0)
            fprintf("Calculating empty FOV row noise-floor values...");
            rowInit = 1;
        end

        if (polarPlotMode == 1)
            displayPolarHeatmap(rangeAzimuth, theta, range);

            displayPolarZones(decisionValue, numZones, zone, occ_color);
        end
        if (polarPlotMode == 2)
            displayRectangleHeatmap(rangeAzimuth, theta_degree, range,fin_clt_frm);

%             displayRectangleZones(decisionValue, numZones, zone, occ_color);
        end

        drawnow;
       
        
%% Phase collection %% 3048 or 3050, 1512

        if  Flag_Beam_Lock == 1 && phase_cntr <=1512
            if target_angle > 0  
                Phase_sig(phase_cntr, 1) = extractedValue(1);
             elseif target_angle < 0
                Phase_sig(phase_cntr, 1) = extractedValue(6);
             else
                Phase_sig(phase_cntr, 1) = (extractedValue(6)+extractedValue(1))/2;
             end               
            phase_cntr = phase_cntr +1;
        elseif phase_cntr > 1512
            %% write the stop stime in file
            fprintf(fid, "The stop Time is: %s\n", datetime(now, "ConvertFrom", "datenum"));
            fclose(fid);
            
            %% write the Phase signal in file
            writematrix(Phase_sig, filename_phase);
            
            %% terminate loop
            data_collection_on=0;       
        end

        
        %% Logging and packet diagnostic 
        byteVecIdx = Header.totalPacketLen;

        if ((Header.frameNumber - packetNumberPrev) ~= 1) && (packetNumberPrev ~= 0)
            fprintf('Error: Packets lost: %d, current frame num = %d \n', (Header.frameNumber - packetNumberPrev - 1), Header.frameNumber)
        end

        packetNumberPrev = Header.frameNumber;
    end  %% magic ok completion read data cycle

    %% Remove processed data
    if byteVecIdx > 0
        shiftSize = byteVecIdx;
        bytevec_cp(1: bytevec_cp_len-shiftSize) = bytevec_cp(shiftSize+1:bytevec_cp_len);
        bytevec_cp_len = bytevec_cp_len - shiftSize;
        if bytevec_cp_len < 0
            fprintf('Error: bytevec_cp_len < bytevecAccLen, %d %d \n', bytevec_cp_len, bytevecAccLen)
            bytevec_cp_len = 0;
        end
    end
    if bytevec_cp_len > (bytevec_cp_max_len * 7/8)
        bytevec_cp_len = 0;
    end

    tIdleStart = tic;

    pause(0.01);


    if(toc(tIdleStart) > 2*Params.frameCfg.framePeriodicity/1000)
        timeout_ctr=timeout_ctr+1;
        if debugFlag == 1
            fprintf('Timeout counter = %d\n', timeout_ctr);
        end
        tIdleStart = tic;
    end
    time_exp=time_exp+1;
end %% data is read adn while loop closed



%% Stop the sensor
cliCfg_stop{1} = 'sensorStop';
sendConfigToTarget(comportCliNum, cliCfg_stop, cliCfgFileName);


%% Close the handles
fclose(sphandle); %close com port
delete(sphandle);


return


function displayPolarHeatmap(rangeAzimuth_2plot, theta, range)
global rollingMax
global rollingAvg
global rollingIdx

figure(1)

heatmapMax = max(rangeAzimuth_2plot(:));

% Create a 3 frame rolling average of the max value
rollingMax(rollingIdx) = heatmapMax;
rollingAvg = mean(rollingMax);
rollingIdx = rollingIdx + 1;
if (rollingIdx == 4)
    rollingIdx = 1;
end

%   cLim = [0, Inf];
if (rollingAvg < 1000)
    cLim = [0, 1000];
else
    cLim = [0, rollingAvg];
end

imagesc_polar2(theta, range, rangeAzimuth_2plot, cLim); hold on
set(gca,'XDir','reverse');

xlabel('Azimuth [m]');
ylabel('Range [m]');
yLim = [0, range(end)];
xLim = yLim(2)*sin(max(abs(theta))) * [-1,1];
ylim(yLim);
xlim(xLim);
delta = 0.5;
set(gca, 'Xtick', [-50:delta:50]);
set(gca, 'Ytick', [0:delta:100]);
set(gca,'Color', [0.5 0.5 0.5])
grid on;
return


function displayRectangleHeatmap(rangeAzimuth_2plot, theta_degree, range,fin_clt_frm)
global rollingMax
global rollingAvg
global rollingIdx
figure(1)

%% If A is a matrix, then max(A) is a row vector containing the maximum value of each column.
%% No reflector between Human and Radar
heatmapMax = max(rangeAzimuth_2plot(:));
% Create a 3 frame rolling average of the max value
Diff_Frame = double(rangeAzimuth_2plot) - double(fin_clt_frm);
rollingMax(rollingIdx) = heatmapMax;
rollingAvg = mean(rollingMax);
rollingIdx = rollingIdx + 1;
if (rollingIdx == 4)
    rollingIdx = 1;
end
if (rollingAvg < 1000)
    cLim = [0, 1000];
else
    cLim = [0, rollingAvg];
end

imagesc(theta_degree, range, Diff_Frame, cLim);
set(gca,'YDir','normal')
set(gca,'XDir','reverse');
xlabel('Azimuth Angle [degree]');
ylabel('Range [m]');


return


function displayPolarZones(decisionValue, numZones, zone, occ_color)

hold on;
for zIdx = 1:numZones
    if decisionValue(zIdx)
        plot(zone(zIdx).boundary.x, zone(zIdx).boundary.y, 'Color', occ_color, 'LineWidth', 2.0);
    else
        plot(zone(zIdx).boundary.x, zone(zIdx).boundary.y, 'Color', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
    end
end
hold off
return


function displayRectangleZones(decisionValue, numZones, zone, occ_color)

hold on;
for zIdx = 1:numZones
    if decisionValue(zIdx)
        rectangle('Position', zone(zIdx).rect, 'EdgeColor', occ_color, 'LineWidth', 2.0);
    else
        rectangle('Position', zone(zIdx).rect, 'EdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
    end
end
hold off
return


%------------------------------------------------------------------------------
function cliCfg = readConfigFile(cliCfgFileName)
%% Read Configuration file
cliCfgFileId = fopen(cliCfgFileName, 'r');
if cliCfgFileId == -1
    fprintf('File %s not found!\n', cliCfgFileName);
    return
else
    fprintf('Opening configuration file %s ...\n', cliCfgFileName);
end
cliCfg = [];
tline = fgetl(cliCfgFileId);
k = 1;
while ischar(tline)
    cliCfg{k} = tline;
    tline = fgetl(cliCfgFileId);
    k = k + 1;
end
fclose(cliCfgFileId);

return

%------------------------------------------------------------------------------
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

%------------------------------------------------------------------------------
function [sphandle] = configureCliPort(comportPnum)
%if ~isempty(instrfind('Type','serial'))
%    disp('Serial port(s) already open. Re-initializing...');
%    delete(instrfind('Type','serial'));  % delete open serial ports.
%end
comportnum_str = ['COM' num2str(comportPnum)]
sphandle = serial(comportnum_str, 'BaudRate', 115200);
set(sphandle, 'Parity', 'none')
set(sphandle, 'Terminator', 'LF')

fopen(sphandle);

return

%------------------------------------------------------------------------------
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

%------------------------------------------------------------------------------
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

%------------------------------------------------------------------------------
function [] = readUartCallbackFcn(obj, event)
global bytevecAcc;
global bytevecAccLen;
global readUartFcnCntr;
global BYTES_AVAILABLE_FLAG
global BYTES_AVAILABLE_FCN_CNT
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

%------------------------------------------------------------------------------
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

%------------------------------------------------------------------------------
function [tlv, idx] = getTlv(bytevec, idx)
word = [1 256 65536 16777216]';
tlv.type = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
tlv.length = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
return

%------------------------------------------------------------------------------
%------------------------------------------------------------------------------
%------------------------------------------------------------------------------
function [rangeAzimuth, idx] = getOccupRangeAzimuthHeatMap(bytevec, idx, numRangeBins, numAngleBins) %==>>
len = numRangeBins * numAngleBins * 4;
rangeAzimuth = bytevec(idx+1:idx+len);
idx = idx + len;

% group 4 bytes typecase to single
rangeAzimuth = typecast(uint8(rangeAzimuth), 'single');
return


%------------------------------------------------------------------------------
function [rangeAzimuth, idx] = getOccupShortHeatMap(bytevec, idx, numRangeBins, numAngleBins) %==>>
len = numRangeBins * numAngleBins * 2;
rangeAzimuth = bytevec(idx+1:idx+len);
idx = idx + len;

% group 2 bytes typecase to single
rangeAzimuth = typecast(uint8(rangeAzimuth), 'uint16');
return


%------------------------------------------------------------------------------
function [rangeAzimuth, idx] = getOccupByteHeatMap(bytevec, idx, numRangeBins, numAngleBins) %==>>
len = numRangeBins * numAngleBins;
rangeAzimuth = bytevec(idx+1:idx+len);
idx = idx + len;

% group 2 bytes typecase to single
rangeAzimuth = typecast(uint8(rangeAzimuth), 'uint8');
return

%------------------------------------------------------------------------------
function [decisionValue, idx] = getOccupDecision(bytevec, idx, numZones) %==>>
%len = 6; % uint8_t
len = numZones;
decisionValue = bytevec(idx+1:idx+len);
idx = idx + len;

% group 4 bytes typecase to single
% decisionValue = typecast(uint8(decisionValue), 'uint32');

return


%------------------------------------------------------------------------------
function [idx] = dumpRowNoiseValues(bytevec, idx, numRangeBins) %==>>
global rowInit;
%    len = 6; % uint8_t
%    idx = idx + len;

rowInit = 2;
len = numRangeBins * 4; % 1 single per row
rowNoise = bytevec(idx+1:idx+len);
idx = idx + len;

% group 4 bytes typecase to single
rowNoise = typecast(uint8(rowNoise), 'single');
fprintf("\n");
row = 1;

for grp = 1:numRangeBins / 8
    fprintf("rowNoise %2d %d ", (grp-1)*8, 8);
    for ridx = 1:8
        fprintf(" %f", rowNoise(row));
        row = row + 1;
    end

    fprintf("\n");
end

return


%------------------------------------------------------------------------------
function [extractedValue, idx] = getVitalSignsHeartBreathingRate(bytevec, idx) %==>>
num_of_outputs = 10;
size_of_float = 4; %32-bit values (4 bytes)
len = size_of_float * num_of_outputs;
extractedValue = bytevec(idx+1:idx+len);
idx = idx + len;

% group 4 bytes typecase to single
extractedValue = typecast(uint8(extractedValue), 'single');
%fprintf("%.2f \n",extractedValue);

return


%------------------------------------------------------------------------------
function [StatsInfo, idx] = getStatsInfo(bytevec, idx)
word = [1 256 65536 16777216]';
StatsInfo.interFrameProcessingTime = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
StatsInfo.transmitOutputTime = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
StatsInfo.interFrameProcessingMargin = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
StatsInfo.interChirpProcessingMargin = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
StatsInfo.activeFrameCPULoad = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
StatsInfo.interFrameCPULoad = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
return


%------------------------------------------------------------------------------
% Read relevant CLI parameters and store into P structure
function [P] = parseConfig(cliCfg)

% global TOTAL_PAYLOAD_SIZE_BYTES
global MAX_NUM_OBJECTS
global OBJ_STRUCT_SIZE_BYTES
global platformType
global STATS_SIZE_BYTES
global rowInit

P=[];
for k = 1:length(cliCfg)
    C = strsplit(cliCfg{k});
    if strcmp(C{1},'channelCfg')
        P.channelCfg.txChannelEn = str2num(C{3});
        if platformType == hex2dec('a1843')
            P.dataPath.numTxAzimAnt = bitand(bitshift(P.channelCfg.txChannelEn,0),1) +...
                bitand(bitshift(P.channelCfg.txChannelEn,-1),1);
            P.dataPath.numTxElevAnt = 0;
        elseif platformType == hex2dec('a1443')
            P.dataPath.numTxAzimAnt = bitand(bitshift(P.channelCfg.txChannelEn,0),1) +...
                bitand(bitshift(P.channelCfg.txChannelEn,-2),1);
            P.dataPath.numTxElevAnt = bitand(bitshift(P.channelCfg.txChannelEn,-1),1);
        else
            fprintf('Unknown platform \n');
            return
        end
        P.channelCfg.rxChannelEn = str2num(C{2});
        P.dataPath.numRxAnt = bitand(bitshift(P.channelCfg.rxChannelEn,0),1) +...
            bitand(bitshift(P.channelCfg.rxChannelEn,-1),1) +...
            bitand(bitshift(P.channelCfg.rxChannelEn,-2),1) +...
            bitand(bitshift(P.channelCfg.rxChannelEn,-3),1);
        P.dataPath.numTxAnt = P.dataPath.numTxElevAnt + P.dataPath.numTxAzimAnt;
    elseif strcmp(C{1},'dataFmt')
    elseif strcmp(C{1},'profileCfg')
        P.profileCfg.startFreq = str2num(C{3});
        P.profileCfg.idleTime =  str2num(C{4});
        P.profileCfg.rampEndTime = str2num(C{6});
        P.profileCfg.freqSlopeConst = str2num(C{9});
        P.profileCfg.numAdcSamples = str2num(C{11});
        P.profileCfg.digOutSampleRate = str2num(C{12}); %uints: ksps
    elseif strcmp(C{1},'chirpCfg')
    elseif strcmp(C{1},'frameCfg')
        P.frameCfg.chirpStartIdx = str2num(C{2});
        P.frameCfg.chirpEndIdx = str2num(C{3});
        P.frameCfg.numLoops = str2num(C{4});
        P.frameCfg.numFrames = str2num(C{5});
        P.frameCfg.framePeriodicity = str2num(C{6});
    elseif strcmp(C{1},'guiMonitor')
        P.guiMonitor.decision = str2num(C{2});
        P.guiMonitor.rangeAzimuthHeatMap = str2num(C{3});
    elseif strcmp(C{1},'zoneDef')
        P.numZones = str2num(C{2});
        cellIdx = 2;
        for z = 1:P.numZones
            P.zoneDef{z} =  [str2num(C{cellIdx+1}), str2num(C{cellIdx+2}), str2num(C{cellIdx+3}), str2num(C{cellIdx+4})];
            cellIdx = cellIdx + 4;
        end
    elseif strcmp(C{1},'coeffMatrixRow')
        pair = str2num(C{2}) + 1;
        rowIdx = str2num(C{3}) + 1;
        P.coeffMatrix(pair, rowIdx, 1:6) =  [str2num(C{4}), str2num(C{5}), str2num(C{6}), str2num(C{7}), str2num(C{8}), str2num(C{9})];

    elseif strcmp(C{1},'meanVector')
        pair = str2num(C{2}) + 1;
        P.meanVector =  [pair, str2num(C{3}), str2num(C{4}), str2num(C{5}), str2num(C{6}), str2num(C{7})];

    elseif strcmp(C{1},'stdVector')
        pair = str2num(C{2}) + 1;
        P.stdVector  =  [pair, str2num(C{3}), str2num(C{4}), str2num(C{5}), str2num(C{6}), str2num(C{7})];
    elseif strcmp(C{1},'oddemoParms')
        P.windowLen  =  str2num(C{2});
        P.diagLoadFactor =  str2num(C{3});
    elseif strcmp(C{1},'rowNoise')
        rowInit = 1;
        frow = str2num(C{2}) + 1;
        cnt  = str2num(C{3});
        fidx = 4;

        for row = frow:(frow+cnt-1)
            P.rowNoise{row} = str2num(C{fidx});
            fidx = fidx + 1;
        end
    end
end
P.dataPath.numChirpsPerFrame = (P.frameCfg.chirpEndIdx -...
    P.frameCfg.chirpStartIdx + 1) *...
    P.frameCfg.numLoops;
P.dataPath.numDopplerBins = P.dataPath.numChirpsPerFrame / P.dataPath.numTxAnt;
P.dataPath.numRangeBins = pow2roundup(P.profileCfg.numAdcSamples);
P.dataPath.rangeResolutionMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
    (2 * P.profileCfg.freqSlopeConst * 1e12 * P.profileCfg.numAdcSamples);
P.dataPath.rangeIdxToMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
    (2 * P.profileCfg.freqSlopeConst * 1e12 * P.dataPath.numRangeBins);
P.dataPath.dopplerResolutionMps = 3e8 / (2*P.profileCfg.startFreq*1e9 *...
    (P.profileCfg.idleTime + P.profileCfg.rampEndTime) *...
    1e-6 * P.dataPath.numDopplerBins * P.dataPath.numTxAnt);

return

%------------------------------------------------------------------------------
function [y] = pow2roundup (x)
y = 1;
while x > y
    y = y * 2;
end
return

%------------------------------------------------------------------------------
function imagesc_polar2(theta, rr, im, cLim) %==>>
% Plot imagesc-like plot in polar coordinates using pcolor()

if nargin<4, cLim = []; end

% transform data in polar coordinates to Cartesian coordinates.
YY = rr'*cos(theta);
XX = rr'*sin(theta);

% plot data on top of grid
h = pcolor(XX, YY, im);
shading flat
grid on;
axis equal;

%
if ~isempty(cLim)
    caxis(cLim);
end

return

%------------------------------------------------------------------------------
function zone = define_zone(rgVal, azVal, def) %==>>
% zoneDef: range_start range_length azimuth_start azimuth_length.
% range_start and azimuth_start index starts from zero

zone.def    = def;
zone.rgIdx  = (zone.def(1)+1:zone.def(1)+zone.def(2));
zone.azIdx  = (zone.def(3)+1:zone.def(3)+zone.def(4));
zone.rect   = [azVal(zone.azIdx(1)), rgVal(zone.rgIdx(1)), ...
    azVal(zone.azIdx(end))-azVal(zone.azIdx(1)), rgVal(zone.rgIdx(end))-rgVal(zone.rgIdx(1))];

% generates a set of points for the boundary of a zone
zone.boundary = gen_zonePoints(zone.rect);

return

%------------------------------------------------------------------------------
function zonePoints = gen_zonePoints(zoneRect) %==>>
% generates a set of points for the boundary of a zone
% to be overlayed on the polar-cordinate plot

% params
theta           = (zoneRect(1):zoneRect(1)+zoneRect(3)).' * pi/180;
rhoInner        = zoneRect(2);
rhoOuter        = (zoneRect(2)+zoneRect(4));

% points
pointsInner.x   = rhoInner*sin(theta);
pointsInner.y   = rhoInner*cos(theta);
pointsOuter.x   = rhoOuter*sin(theta);
pointsOuter.y   = rhoOuter*cos(theta);

% output
zonePoints.x    = [pointsInner.x; flipud(pointsOuter.x); pointsInner.x(1)];
zonePoints.y    = [pointsInner.y; flipud(pointsOuter.y); pointsInner.y(1)];

return

%------------------------------------------------------------------------------
function g = sigmoid(z) %==>>
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = 1 ./ (1 + exp(-z));

return

%------------------------------------------------------------------------------
function [occupVec, featureVec] = zone_occupDetect(rangeAzimuth, coeffMatrix, meanVector, ...  %==>>
    stdVector, frameIdx, winLen, zone, numZones)
global zonePwr
global zonePwrdB

avgPwr      = zeros(1, numZones);
avgPwrdB    = zeros(1, numZones);

% zone-power in each frame
cirBufferIdx = mod(frameIdx-1, winLen) + 1;

% moving window index: newest sample at the end
winIdx = mod((cirBufferIdx-winLen:cirBufferIdx-1), winLen) + 1;

for zIdx = 1:numZones
    % calculate zone power
    rangeAzimuth_rgAzGated          = rangeAzimuth(zone(zIdx).rgIdx, zone(zIdx).azIdx);
    zonePwr(cirBufferIdx, zIdx)     = mean(rangeAzimuth_rgAzGated(:));
    zonePwrdB(cirBufferIdx, zIdx)   = 10*log10(zonePwr(cirBufferIdx, zIdx));

    % features: moving average of zonePwr in dB
    avgPwr(zIdx)                    = mean(zonePwr(winIdx, zIdx));
    avgPwrdB(zIdx)                  = 10*log10( avgPwr(zIdx) );
end

% features: power ratio in dB
pwrRatio    = avgPwr / sum(avgPwr);
pwrRatiodB  = 10*log10(pwrRatio);

% features: correlation coefficient between pairs
corrCoeff   = corrcoef(zonePwrdB(winIdx, :));
xcorrCoeff  = corrCoeff(1, 2);

% form the feature vector
featureVec  = [avgPwrdB, pwrRatiodB, xcorrCoeff]; % 1 x 5  for two zones

% normalize and add one
featureVec  = (featureVec - meanVector) ./ stdVector;
featureVec_  = [1, featureVec].'; % now column vector

% occupancy detection
prob                = sigmoid(coeffMatrix * featureVec_);
[~, class_predict]  = max(prob);
class_predict       = class_predict - 1;
occupVec            = de2bi(class_predict, numZones);

return
%---------------------------------------------------------------------------------
