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