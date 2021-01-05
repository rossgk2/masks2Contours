function insert_indices = getRVinserts(points)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function gets the indices of the RV insert points from the end
% points of the RV free wall contour points
%
% Renee Miller (renee.miller@kcl.ac.uk)
% Date: 5 February 2019
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialise list of distances between adjacent points
d = zeros(length(points),1);

% Loop through each point
for i = 1:length(points)
    % Get distance between neighboring points
    if i == length(points)
        pts = [points(i,:); points(1,:)]; % Compare last point with the first point
        d(i) = pdist(pts,'euclidean');
    else
        d(i) = pdist(points(i:i+1,:),'euclidean');
    end
end

% Get outlying points (largest distance between two consecutive points)
[~, ~, u, ~] = isoutlier(d, 'mean');

% Check to see if they are larger than the mean distance
if ~isempty(find(d > u))
    [~, md] = max(d);
    if md == length(d)
        insert_indices = [1; md];
        disp("insert_indices (case if-if)");
        disp(insert_indices);
    else
        insert_indices = [md; md+1];
        disp("insert_indices (case if-else)");
        disp(insert_indices);
    end
else
    insert_indices = [];
    disp("case else-else");
end
