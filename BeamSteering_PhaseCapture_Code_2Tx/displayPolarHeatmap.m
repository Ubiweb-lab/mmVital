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