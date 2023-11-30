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
