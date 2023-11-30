function displayRectangleHeatmap(rangeAzimuth_2plot, theta_degree, range)
global rollingMax
global rollingAvg
global rollingIdx
figure(1)

%% If A is a matrix, then max(A) is a row vector containing the maximum value of each column.
%% No reflector between Human and Radar
heatmapMax = max(rangeAzimuth_2plot(:));
% Create a 3 frame rolling average of the max value

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

imagesc(theta_degree, range, rangeAzimuth_2plot, cLim);
set(gca,'YDir','normal')
set(gca,'XDir','reverse');
xlabel('Azimuth Angle [degree]');
ylabel('Range [m]');


return