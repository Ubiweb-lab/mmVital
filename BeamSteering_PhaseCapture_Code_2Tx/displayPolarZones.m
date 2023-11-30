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
