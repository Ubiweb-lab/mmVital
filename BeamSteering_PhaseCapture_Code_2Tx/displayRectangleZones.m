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
