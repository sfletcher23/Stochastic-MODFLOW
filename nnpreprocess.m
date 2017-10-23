%% Neural Network for MODFLOW data

% Current input data series:
%   Horizontal hydraulic conductivity
%   Vertical hydraulic conducitivty 
%   Time
%    
% Current output data series:
%   Head in pumping wells 
% 

%% Combine exisitng .mat files from simulation

runsToUse = 20; 
maxDrawdownRuns = 1;

% Load head data
timeToOpen = '2017-08-15 14:31:17';
headData = [];
for i = 10:11
    filename = strcat('modflowData_headData',num2str(i), timeToOpen,'.mat');
    data = load(filename);
    headDataTemp = data.headData;
    headData = cat(3, headData, headDataTemp);
    clear data headDataTemp
end
headData = headData(:,:,1:runsToUse);

% Load hk and sy data
filename3 = strcat('modflowData_hk',timeToOpen,'.mat');
filename4 = strcat('modflowData_sy',timeToOpen,'.mat');
data = load(filename3);
hk = data.hk;
clear data
data = load(filename4);
sy = data.sy;
clear data
% Get max drawdown samples, then remaining random
hk_sorted = sort(hk);
sy_sorted = sort(sy);
hk_sorted = hk_sorted(1:maxDrawdownRuns);
sy_sorted = sy_sorted(1:maxDrawdownRuns);
hk = [hk(1:runsToUse - maxDrawdownRuns) hk_sorted];
sy = [sy(1:runsToUse - maxDrawdownRuns) sy_sorted];


disp('data loaded')

%% Reduce time granularity, extra high drawdown samples

[numWells, numTime, numRuns] = size(headData);
index = 1:30:numTime;
headData = headData(:,index,:);
[~, numTime, ~] = size(headData);

%
%% Aggregate data for time-series neural net model 

outputs = zeros(numRuns * numTime, numWells);

% Rehape the output data to have all the data for parameter 1, then all
% the data for parameter 2, etc with one time series listed below another.
% Number of wells is the number of columns
tempHeadData = headData;
tempHeadData = permute(headData,[2 3 1]);
outputs = reshape(tempHeadData, [numRuns*numTime,numWells]);
clear tempHeadData headData

inputs = zeros(numRuns * numTime, 3);

% Replicate each static variable so the same value repeats for each time

% period
inputs(:,1) = reshape(repmat(hk(1:numRuns), [numTime,1]),[],1);
inputs(:,2) = reshape(repmat(sy(1:numRuns), [numTime,1]),[],1);

% Reshape time to get a vector repeats each time numRuns times, then des
% the same for the next time value
time = 1:numTime;
inputs(:,3) = repmat(time', [numRuns, 1]);

disp('inputs and outputs created')