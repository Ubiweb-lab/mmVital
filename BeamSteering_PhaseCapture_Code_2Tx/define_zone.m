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