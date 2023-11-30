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