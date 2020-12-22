%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script loads masks (nifti format) and converts them to contour points
% which can be used to fit a finite element model to the data
%
% Written by: Renee Miller (renee.miller@kcl.ac.uk)
% Date moified: 30 October 2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
close all
clear all

% Add all subfolders to current path
addpath('C:\Users\Ross\Documents\MATLAB\LineCurvature');

% Set to 1 to create intermediate plots
PLOT = 1;

% Orientation for viewing plots
az = 214.5268;
el = -56.0884;

% Image directory
%fldr = 'C:\Users\rmi18\Documents\Data\CMR\HCM_Images\Case1'; % Replace with path to folder where your images and labels are located
fldr = 'C:\Users\Ross\Documents\Data\CMR\Student_Project\P3';

% Frame segmented
frame = 1;

% Create results directory
resultsDir = [fldr, '/out'];
mkdir(resultsDir)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SHORT AXIS

% Change these as needed
%imgName = [fldr, '/SA.nii']; % Image name
imgName = [fldr, '/CINE_SAX.nii']; % Image name

%segName = [fldr, '/SA_FR', num2str(frame), '_RM.nii']; % Label name
segName = [fldr, '/CINE_SAX_', num2str(frame), '.nii']; % Label name

% Run main function to get contour points from masks - output
% will be saved in mat files in the results directory
masks2contoursSA_manual(segName, imgName, resultsDir, frame, PLOT);

% Load results
load(sprintf('%s/SA_contour_points_FR%d.mat', resultsDir, frame))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LONG AXIS

LA_segs = dir([fldr, '/LAX_*_', num2str(frame), '.nii']); % Change as needed
LA_segs = {LA_segs.name};
LA_segs = strcat(fldr, '/', LA_segs);

LA_names = dir([fldr, '/RReg_LAX_*.nii']); % Change as needed
LA_names = {LA_names.name};
% ind = find(contains(LA_names, num2str(frame)));
% LA_segs = LA_names(ind);
% LA_names( ismember(LA_names, LA_segs) ) = []; 
% LA_segs = strcat(fldr, '/', LA_segs);
LA_names = strcat(fldr, '/', LA_names);


% Run main function to get contour points from masks - output
% will be saved in mat files in the results directory
masks2contoursLA_manual(LA_names, LA_segs, resultsDir, frame, PLOT)

% Load results
load(sprintf('%s/LA_contour_points_FR%d.mat', resultsDir, frame))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Valve points

% Get number of frames
tmp = niftiread(imgName);
numFrames = size(tmp,4);

[mv, tv, av, pv] = compileManualValvePoints_manual(fldr, numFrames, frame); % From manual annotation OR predicted points

mv = reshape(mv, [], 3);
mv = mv(any(mv,2),:);
tv = tv(any(tv,2),:);
av = av(any(av,2),:);
pv = pv(any(pv,2),:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Omit short axis slices without any contours

slices = size(endoLVContours,3);
slices2omit = ~squeeze(endoLVContours(1,1,:));
sliceNums = linspace(1,slices,slices);
sliceNums(slices2omit) = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FIND APEX

% Choose method for calculating apical point
m = 2;

if m == 1
    % Method 1 - uses long axis vector (fit to midpoints of SA centroids) and
    % its intersection with 4CH long axis contour
    % Works best when long axis slices were not well planned
    allLA = permute(epiLVContoursLA, [3 1 2]);
    allLA = reshape(allLA, [], 3);
    tmp = endoLVContours(:,:,max(sliceNums));
    while sum(tmp(:)) == 0
        sliceNumsTmp = setdiff(sliceNums, max(sliceNums));
        tmp = endoLVContours(:,:,max(sliceNumsTmp));
    end
    tmp( ~any(tmp,2), : ) = [];
    
    tmp2 = endoLVContours(:,:,min(sliceNums));
    while sum(tmp2(:)) == 0
        sliceNumsTmp = setdiff(sliceNums, min(sliceNums));
        tmp2 = endoLVContours(:,:,min(sliceNumsTmp));
    end
    tmp2( ~any(tmp2,2), : ) = [];
    tmp_apex = calcApexLA([mean(tmp); mean(tmp2)], allLA, 'debug');
    APEX = (tmp_apex(1:3))';
    
else
    % Method 2 - uses intersection of two long-axis planes
    % This method works best when long axis slices were well-planned
    tmp_apex = calcApex(squeeze(epiLVContoursLA(:,:,1)), squeeze(epiLVContoursLA(:,:,3)), 'debug');
    APEX = (tmp_apex(1:3))';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot all contours points together

FH = figure('position', [50 50 1000 800]);
% Plot short axis contours
for s = 1:length(sliceNums)
    
    i = sliceNums(s);
    
    lv_endo = squeeze(endoLVContours(:,:,i));
    lv_endo = lv_endo(any(lv_endo,2),:);
    
    lv_epi = squeeze(epiLVContours(:,:,i));
    lv_epi = lv_epi(any(lv_epi,2),:);
    
    rv_endo = squeeze(endoRVFWContours(:,:,i));
    rv_endo = rv_endo(any(rv_endo,2),:);
    
    rv_epi = squeeze(epiRVFWContours(:,:,i));
    rv_epi = rv_epi(any(rv_epi,2),:);
    
    rvs = squeeze(RVSContours(:,:,i));
    rvs = rvs(any(rvs,2),:);
    
    rvi = squeeze(RVInserts(:,:,i));
    rvi = rvi(any(rvi,2),:);
    
    h1 = scatter3(lv_endo(:,1), lv_endo(:,2), lv_endo(:,3), 100, 'g.');
    hold on
    h2 = scatter3(lv_epi(:,1), lv_epi(:,2), lv_epi(:,3), 100, 'b.');
    h3 = scatter3(rvs(:,1), rvs(:,2), rvs(:,3), 100, 'y.');
    h4 = scatter3(rv_endo(:,1), rv_endo(:,2), rv_endo(:,3), 100, 'r.');
    h5 = scatter3(rv_epi(:,1), rv_epi(:,2), rv_epi(:,3), 100, 'b.');
    h6 = scatter3(rvi(:,1), rvi(:,2), rvi(:,3), 100, 'ro', 'filled');
end

% Plot valve points
h7 = scatter3(mv(:,1), mv(:,2), mv(:,3), 100, 'ko', 'filled');
hold on
h8 = scatter3(av(:,1), av(:,2), av(:,3), 100, 'co', 'filled');
h9 = scatter3(tv(:,1), tv(:,2), tv(:,3), 100, 'mo', 'filled');

% Plot the apex
a = APEX;
a(~any(a,2),:) = [];
h10 = scatter3(a(1), a(2), a(3), 100, 'ko', 'filled');

% Plot long axis contours
for i = 1:length(LA_segs)
    
    lv_endo = squeeze(endoLVContoursLA(:,:,i));
    lv_endo = lv_endo(any(lv_endo,2),:);
    
    lv_epi = squeeze(epiLVContoursLA(:,:,i));
    lv_epi = lv_epi(any(lv_epi,2),:);
    
    rv_endo = squeeze(endoRVFWContoursLA(:,:,i));
    rv_endo = rv_endo(any(rv_endo,2),:);
    
    rv_epi = squeeze(epiRVFWContoursLA(:,:,i));
    rv_epi = rv_epi(any(rv_epi,2),:);
    
    rvs = squeeze(RVSContoursLA(:,:,i));
    rvs = rvs(any(rvs,2),:);
    
    scatter3(lv_endo(:,1), lv_endo(:,2), lv_endo(:,3), 100, 'g.')
    hold on
    scatter3(lv_epi(:,1), lv_epi(:,2), lv_epi(:,3), 100, 'b.')
    scatter3(rv_endo(:,1), rv_endo(:,2), rv_endo(:,3), 100, 'r.')
    scatter3(rvs(:,1), rvs(:,2), rvs(:,3), 100, 'y.')
    scatter3(rv_epi(:,1), rv_epi(:,2), rv_epi(:,3), 100, 'b.')
    
end

% Plot vector from apex to base
%quiver3(APEX(1), APEX(2), APEX(3), la_vec(1), la_vec(2), la_vec(3), 'AutoScaleFactor', 100, 'LineWidth', 5);

legend([h1, h2, h3, h4, h5, h6, h7, h8, h9, h10], 'LV Endo', 'Epi', 'RVS', ...
    'RVFW Endo', 'RVFW Epi', 'RV Inserts', 'Mitral Valve', 'Aortic Valve', ...
    'Tricuspid Valve', 'Apex', 'FontSize', 12)

xlabel('X', 'FontSize', 12)
ylabel('Y', 'FontSize', 12)
zlabel('Z', 'FontSize', 12)

set(gca, 'FontSize', 12)
view([az el])
xlim([20 120])
ylim([-80 0])
zlim([-40 40])
axis equal
axis off

cs = strsplit(fldr, '\');
title(sprintf('%s: Frame %d', cs{end}, frame))



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Write out contour points to text file for fitting

% File
tmp = strsplit(fldr, '\');
fname = sprintf('%s/GPFile.txt', fldr);
fname1 = sprintf('%s/Case1_FR1.txt', fldr);
fid = fopen(fname, 'w');
fid1 = fopen(fname1, 'w');
fprintf(fid, 'x\ty\tz\tcontour type\tslice\tweight\ttime frame\n');

% Write out short axis contours
for i = 1:length(sliceNums)
    
    % Current slice number
    num = sliceNums(i);
    
    % LV Endocardium
    lv_endo = squeeze(endoLVContours(:,:,num));
    lv_endo = lv_endo(any(lv_endo,2),:);
    
    for j = 1:size(lv_endo,1)
        fprintf(fid1, '%.6f %.6f %.6f saendocardialContour %d %.4f\n', lv_endo(j,:), i, 1);
        fprintf(fid, '%.6f\t%.6f\t%.6f\tSAX_LV_ENDOCARDIAL\t%d\t%.4f\t%d\n', lv_endo(j,:), i, 1, frame);
    end
    
    %RV Free Wall
    rv_endo = squeeze(endoRVFWContours(:,:,num));
    rv_endo = rv_endo(any(rv_endo,2),:);
    
    for j = 1:size(rv_endo,1)
        fprintf(fid1, '%.6f %.6f %.6f RVFW %d %.4f\n', rv_endo(j,:), i, 1);
        fprintf(fid, '%.6f\t%.6f\t%.6f\tSAX_RV_FREEWALL\t%d\t%.4f\t%d\n', rv_endo(j,:), i, 1, frame);
    end
    
    % Epicardium
    lv_epi = squeeze(epiLVContours(:,:,num));
    lv_epi = lv_epi(any(lv_epi,2),:);
    rv_epi = squeeze(epiRVFWContours(:,:,num));
    rv_epi = rv_epi(any(rv_epi,2),:);
    epi = [lv_epi; rv_epi];
    
    for j = 1:size(epi,1)
        fprintf(fid1, '%.6f %.6f %.6f saepicardialContour %d %.4f\n', epi(j,:), i, 1);
        fprintf(fid, '%.6f\t%.6f\t%.6f\tSAX_LV_EPICARDIAL\t%d\t%.4f\t%d\n', epi(j,:), i, 1, frame);
    end
    
    % RV Septum
    rvs = squeeze(RVSContours(:,:,num));
    rvs = rvs(any(rvs,2),:);
    
    for j = 1:size(rvs,1)
        fprintf(fid1, '%.6f %.6f %.6f RVS %d %.4f\n', rvs(j,:), i, 1);
        fprintf(fid, '%.6f\t%.6f\t%.6f\tSAX_RV_SEPTUM\t%d\t%.4f\t%d\n', rvs(j,:), i, 1, frame);
    end
    
    % RV inserts
    rvi = squeeze(RVInserts(:,:,num));
    rvi = rvi(any(rvi,2),:);
    
    for j = 1:size(rvi,1)
        fprintf(fid1, '%.6f %.6f %.6f RV_insert %d %.4f\n', rvi(j,:), i, RVInsertsWeights(num));
        fprintf(fid, '%.6f\t%.6f\t%.6f\tRV_INSERT\t%d\t%.4f\t%d\n', rvi(j,:), i, RVInsertsWeights(num), frame);
    end
    
end

% Save final slice num (keep counting from here for LA slices)
finalSASliceNum = i;

% Write out long axis contours
for i = 1:length(LA_segs)
    
    tmp_name = strsplit(LA_segs{i}, '/');
    tmp_name = tmp_name{end};
    
    % LV Endocardium
    lv_endo = squeeze(endoLVContoursLA(:,:,i));
    lv_endo = lv_endo(any(lv_endo,2),:);
    
    for j = 1:size(lv_endo,1)
        fprintf(fid1, '%.6f %.6f %.6f saendocardialContour %d %.4f\n', lv_endo(j,:), i+finalSASliceNum, 1);
        fprintf(fid, '%.6f\t%.6f\t%.6f\tLAX_LV_ENDOCARDIAL\t%d\t%.4f\t%d\n', lv_endo(j,:), i+finalSASliceNum, 1, frame);
    end
    
    % RV Free Wall
    rv_endo = squeeze(endoRVFWContoursLA(:,:,i));
    rv_endo = rv_endo(any(rv_endo,2),:);
    
    for j = 1:size(rv_endo,1)
        fprintf(fid1, '%.6f %.6f %.6f RVFW %d %.4f\n', rv_endo(j,:), i+finalSASliceNum, 1);
        fprintf(fid, '%.6f\t%.6f\t%.6f\tLAX_RV_FREEWALL\t%d\t%.4f\t%d\n', rv_endo(j,:), i+finalSASliceNum, 1, frame);
    end
    
    % Epicardium
    lv_epi = squeeze(epiLVContoursLA(:,:,i));
    lv_epi = lv_epi(any(lv_epi,2),:);
    rv_epi = squeeze(epiRVFWContoursLA(:,:,i));
    rv_epi = rv_epi(any(rv_epi,2),:);
    epi = [lv_epi; rv_epi];
    
    for j = 1:size(epi,1)
        fprintf(fid1, '%.6f %.6f %.6f saepicardialContour %d %.4f\n', epi(j,:), i+finalSASliceNum, 1);
        fprintf(fid, '%.6f\t%.6f\t%.6f\tLAX_LV_EPICARDIAL\t%d\t%.4f\t%d\n', epi(j,:), i+finalSASliceNum, 1, frame);
    end
    
    % RV Septum
    rvs = squeeze(RVSContoursLA(:,:,i));
    rvs = rvs(any(rvs,2),:);
    
    if ~isempty(rvs)
        for j = 1:size(rvs,1)
            fprintf(fid1, '%.6f %.6f %.6f RVS %d %.4f\n', rvs(j,:), i+finalSASliceNum, 1);
            fprintf(fid, '%.6f\t%.6f\t%.6f\tLAX_RV_SEPTUM\t%d\t%.4f\t%d\n', rvs(j,:), i+finalSASliceNum, 1, frame);
        end
    end
    
    % Tricuspid Valve
    if ~isempty(tv) && contains(tmp_name, '4') 
        for j = 1:size(tv,1)
            fprintf(fid1, '%.6f %.6f %.6f Tricuspid_Valve %d %.4f\n', tv(j,:), i+finalSASliceNum, 1);
            fprintf(fid, '%.6f\t%.6f\t%.6f\tTRICUSPID_VALVE\t%d\t%.4f\t%d\n', tv(j,:), i+finalSASliceNum, 1, frame);
        end
    end
    
    % Mitral Valve
    if ~isempty(mv)
        for j = 1:2
            fprintf(fid1, '%.6f %.6f %.6f BP_point %d %.4f\n', mv(j,:), i+finalSASliceNum, 1);
            fprintf(fid, '%.6f\t%.6f\t%.6f\tMITRAL_VALVE\t%d\t%.4f\t%d\n', mv(j,:), i+finalSASliceNum, 1, frame);
        end
        mv(1:2,:) = [];
    end
    
    % Aortic valve
    if ~isempty(av) && (contains(tmp_name, '3') ||  contains(tmp_name, 'LVOT'))
        for j = 1:size(av,1)
            fprintf(fid1, '%.6f %.6f %.6f Aorta %d %.4f\n', av(j,:), i+finalSASliceNum, 1);
            fprintf(fid, '%.6f\t%.6f\t%.6f\tAORTA_VALVE\t%d\t%.4f\t%d\n', av(j,:), i+finalSASliceNum, 1, frame);
        end
    end
    
    % Pulmonary valve
    if ~isempty(pv) && contains(tmp_name, 'RVOT')
        for j = 1:size(pv,1)
            fprintf(fid1, '%.6f %.6f %.6f Pulmonary %d %.4f\n', av(j,:), i+finalSASliceNum, 1);
            fprintf(fid, '%.6f\t%.6f\t%.6f\tPULMONARY_VALVE\t%d\t%.4f\t%d\n', av(j,:), i+finalSASliceNum, 1, frame);
        end
    end
end

% Apex
fprintf(fid1, '%.6f %.6f %.6f LA_Apex_Point %d %.4f\n', APEX, i+finalSASliceNum, 1);
fprintf(fid, '%.6f\t%.6f\t%.6f\tAPEX_POINT\t%d\t%.4f\t%d\n', APEX, i+finalSASliceNum, 1, frame);

fclose(fid);

