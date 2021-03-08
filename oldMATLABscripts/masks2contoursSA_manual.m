function masks2contoursSA_manual(segName, imgName, resultsDir, frameNum, PLOT)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script loads masks (nifti format) and converts them to contour points
% which can be used to fit a finite element model to the data
%
% Inputs: 1) segName - file name of segmentation/label
%         2) imgName - file name of the corresponding image set
%         3) resultsDir - directory where to save results
%         4) frameNum - frame number (time point in cardiac cycle)
%         5) PLOT - set to 0 (don't create plots) or 1 (create intermediate
%         plots)
%
% Output is written to a mat file
%
% Written by: Renee Miller (renee.miller@kcl.ac.uk)
% Date moified: 4 February 2019
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set RV wall thickness (don't have contours)
rv_wall = 3; % mm

% Down sampling contour points - how much to downsample by (e.g. 3 ==> take
% every third point)
ds = 3;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SHORT AXIS

% Load short axis segmentations and header info
seg = niftiread(segName);
if length(size(seg))>3 % If segmentation includes all time points
    seg = squeeze(seg(:,:,:,frameNum));
end
info = niftiinfo(imgName);

% Transformation matrix
transform = info.Transform.T';
transform(1:2,:) = transform(1:2,:)*-1; % This edit has to do with RAS system in Nifti files

% UPDATE
if norm(info.Transform.T(1:3,1)) == 1
    pix_scale = info.PixelDimensions(1:3)';
else
    pix_scale = [1 1 1]';
end

% pixel spacing
pixSpacing = info.PixelDimensions(1);

% Number of slices
slices = size(seg,3);

% Separate out LV endo, LV epi and RV endo segmentations
endoLV = seg==1;
epiLV = double(seg<=2).*double(seg>0);
endoRV = seg==3;

% Initialise variables
endoLVContours = zeros(200,3,slices);
epiLVContours = zeros(200,3,slices);
endoRVFWContours = zeros(200,3,slices);
epiRVFWContours = zeros(200,3,slices);
RVSContours = zeros(200,3,slices);
RVInserts = zeros(2,3,slices);

%For debug
removedPointsAll = [];

mkdir(sprintf("%s\\endoLVmask_slices", resultsDir))
mkdir(sprintf("%s\\epiLVmask_slices", resultsDir))
mkdir(sprintf("%s\\endoRVmask_slices", resultsDir))
mkdir(sprintf("%s\\tmp_endoLV_slices", resultsDir))
mkdir(sprintf("%s\\tmp_epiLV_slices", resultsDir))
mkdir(sprintf("%s\\tmp_endoRV_slices", resultsDir))
mkdir(sprintf("%s\\tmp_RVFW_slices", resultsDir))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loop through short axis slices
for i = 1:slices
    
    % Get masks for current slice
    % bwareaopen - gets rid of any random pixels which have somehow made
    % their way into the the segmentation, area must contain 50 pixels or
    % more; imfill gets rid of any holes in the segmentation area
    endoLVmask = imfill(double(bwareaopen(squeeze(endoLV(:,:,i)),50)));
    epiLVmask = imfill(double(bwareaopen(squeeze(epiLV(:,:,i)),50)));
    endoRVmask = imfill(double(bwareaopen(squeeze(endoRV(:,:,i)),50)));
    
    % Get contours from masks
    tmp_endoLV = mask2poly(endoLVmask, 'Exact', 'CW');
    tmp_epiLV = mask2poly(epiLVmask, 'Exact', 'CW');
    tmp_endoRV = mask2poly(endoRVmask, 'Exact', 'CW');    
    
    % Save the slice data, for debug purposes
    save(sprintf("%s\\endoLVmask_slices\\endoLVmask_slice_%d.mat", resultsDir, i), "endoLVmask")
    save(sprintf("%s\\epiLVmask_slices\\epiLVmask_slice_%d.mat", resultsDir, i), "epiLVmask")
    save(sprintf("%s\\endoRVmask_slices\\endoRVmask_slice_%d.mat", resultsDir, i), "endoRVmask")
    
    save(sprintf("%s\\tmp_endoLV_slices\\tmp_endoLV_slice_%d.mat", resultsDir, i), "tmp_endoLV")
    save(sprintf("%s\\tmp_epiLV_slices\\tmp_epiLV_slice_%d.mat", resultsDir, i), "tmp_epiLV")
    save(sprintf("%s\\tmp_endoRV_slices\\tmp_endoRV_slice_%d.mat", resultsDir, i), "tmp_endoRV")
    
    % Differentiate contours for RVFW (free wall) and RVS (septum)
    [tmp_RVS, ia, ib] = intersect(tmp_epiLV, tmp_endoRV, 'rows');
    tmp_RVFW = tmp_endoRV;
    tmp_RVFW(ib,:) = [];
    
    save(sprintf("%s\\tmp_RVFW_slices\\tmp_RVFW_slice_%d.mat", resultsDir, i), "tmp_RVFW")

    
    % Remove RVS contour points from LV epi contour
    tmp_epiLV(ia,:) = [];
    
    if PLOT == 1
        % Plot the mask with the contours (to check)
        FH = figure('position', [100 100 1600 500]);
    end
    
    %% LV endo
    if ~isempty(tmp_endoLV)
        
        tmp_endoLV(tmp_endoLV(:,1)<0,:) = []; % Remove any points which lie outside of image
        tmp_endoLV = tmp_endoLV(1:ds:end,:); % down sample
        [tmp_endoLV, removedPoints] = removePoints(tmp_endoLV); % Check for outlying points - probably due to holes in segmentation
        
        removedPointsAll = [removedPointsAll; removedPoints];
        
        if PLOT == 1
            % Plot the mask and contour
            subplot(1,3,1)
            imagesc(squeeze(endoLV(:,:,i)))
            hold on
            colormap gray
            scatter(tmp_endoLV(:,1), tmp_endoLV(:,2), 'ro', 'filled')
            title('LV Endocardium')
            axis image
        end
        
        % Convert contours to image coordinate system
        for j = 1:size(tmp_endoLV,1)
            pix = [tmp_endoLV(j,2); tmp_endoLV(j,1); i-1] .* pix_scale;
            tmp = transform * [pix; 1];
            endoLVContours(j,:,i) = (tmp(1:3))';
        end
        
    end
    
    %% LV epi
    if ~isempty(tmp_epiLV)
        
        tmp_epiLV(tmp_epiLV(:,1)<0,:) = []; % Remove any points which lie outside of image
        tmp_epiLV = tmp_epiLV(1:ds:end,:); % down sample
        [tmp_epiLV, ~] = removePoints(tmp_epiLV); % Check for outlying points - probably due to holes in segmentation
        
        if PLOT == 1
            % Plot the mask and contour
            subplot(1,3,2)
            imagesc(squeeze(epiLV(:,:,i)))
            hold on
            colormap gray
            scatter(tmp_epiLV(:,1), tmp_epiLV(:,2), 'bo', 'filled')
            title('LV Epicardium')
            axis image
        end
        
        % Convert contours to image coordinate system
        for j = 1:size(tmp_epiLV,1)
            pix = [tmp_epiLV(j,2); tmp_epiLV(j,1); i-1] .* pix_scale;
            tmp = transform * [pix; 1];
            epiLVContours(j,:,i) = (tmp(1:3))';
        end
        
    end
    
    %% RV
    if size(tmp_RVFW,1) > 6 %~isempty(tmp_RVFW)
        
        tmp_RVFW(tmp_RVFW(:,1)<0,:) = []; % Remove any points which lie outside of image
        tmp_RVFW = tmp_RVFW(1:ds:end,:); % down sample
        [tmp_RVFW, ~] = removePoints(tmp_RVFW); % Check for outlying points - probably due to holes in segmentation
        
        fprintf('i = %d\n', i);
        tmp_rvi_indices = getRVinserts(tmp_RVFW); % Get the RV insert points
        
        tmp_RVS(tmp_RVS(:,1)<0,:) = []; % Remove any points which lie outside of image
        tmp_RVS = tmp_RVS(1:ds:end,:); % down sample
        if length(tmp_RVS) > 2
            [tmp_RVS, ~] = removePoints(tmp_RVS); % Check for outlying points - probably due to holes in segmentation
        end
        
        if PLOT == 1
            % Plot the mask and contour
            subplot(1,3,3)
            imagesc(squeeze(endoRV(:,:,i)))
            hold on
            colormap gray
            scatter(tmp_RVFW(:,1), tmp_RVFW(:,2), 'go', 'filled')
            scatter(tmp_RVS(:,1), tmp_RVS(:,2), 'yo', 'filled')
            title('RV Endocardium')
            axis image
        end
        
        % Convert contours to image coordinate system
        for j = 1:size(tmp_RVS,1)
            pix = [tmp_RVS(j,2); tmp_RVS(j,1); i-1] .* pix_scale;
            tmp = transform * [pix; 1];
            RVSContours(j,:,i) = (tmp(1:3))';
        end
        
        % Convert contours to image coordinate system
        for j = 1:size(tmp_RVFW,1)
            pix = [tmp_RVFW(j,2); tmp_RVFW(j,1); i-1] .* pix_scale;
            tmp = transform * [pix; 1];
            endoRVFWContours(j,:,i) = (tmp(1:3))';
        end
        
        % Save RV insert coordinates
        if ~isempty(tmp_rvi_indices)
            RVInserts(:,:,i) = endoRVFWContours(tmp_rvi_indices(1:2),:,i);
        end
        
        %% Calculate RV epicardial wall by applying a thickness to RV endocardium
        
        % Get normals of RV endocardium
        N = LineNormals2D(tmp_RVFW);
        
        % RV epicardium
        tmp_epiRV = tmp_RVFW + N*ceil(rv_wall/pixSpacing)*-1;
        
        % Convert contours to image coordinate system
        for j = 1:size(tmp_epiRV,1)
            pix = [tmp_epiRV(j,2); tmp_epiRV(j,1); i-1] .* pix_scale;
            tmp = transform * [pix; 1];
            epiRVFWContours(j,:,i) = (tmp(1:3))';
        end
        
    end
    
    if PLOT == 1
        saveas(FH, sprintf('%s/contours_slice%d.png', resultsDir, i));
        close(FH)
    end
    
    %clearvars tmp*
end

save(sprintf('%s/removedPointsAll.mat', resultsDir), 'removedPointsAll')
save(sprintf('%s/endoRVFWContours.mat', resultsDir), 'endoRVFWContours')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate weights for RV insertion points

% Initialise variable to store error
RVInsertsWeights = zeros(2,slices);

% Loop through anterior (1) and posterior (2) sides for fitting a line
for i = 1:2
    
    inserts1 = squeeze(RVInserts(i,:,:))';
    inserts2 = linspace(1,slices,slices)';
    inserts = [inserts1 inserts2];
    
    inserts = inserts(any(inserts(:,1:3),2),:); % Get rid of rows with zeros
    
    points = inserts(:,1:3);
    indices = inserts(:,4);
    
    % Sort RV insert points by error
    [~,err] = fitLine3D(points);
    
    % Save normalized error
    RVInsertsWeights(i,indices) = abs(err)/max(abs(err));
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save results
save(sprintf('%s/SA_contour_points_FR%d.mat', resultsDir, frameNum), 'endoLVContours', 'epiLVContours', 'endoRVFWContours', ...
    'epiRVFWContours', 'RVSContours', 'RVInserts', 'RVInsertsWeights');


