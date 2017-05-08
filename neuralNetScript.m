%% Neural Network for MODFLOW data

% Current input data series:
%   Pumping rate well 1
%   Pumping rate well 2
%   Horizontal hydraulic conductivity
%   Vertical hydraulic conducitivty 
%   Time
%    
% Current output data series:
%   Head in pumping well 1
%   Head in pumping well 2
%
% Potential future input data series:
%   Storativity
%   Specific yield
%   Aquifer depth
% 

%% Combine exisitng .mat files from simulation

% Get name and number of data files
filenames = dir('*modflowData*.mat');
numFiles = length(filenames(not([filenames.isdir])));

head_data1_overTime = [];
head_data2_overTime = [];
hk = [];
vka = [];
pump_rate1 = [];
pump_rate2 = [];

% Get data as arrays of time
for i = 1:numFiles
    
    % Load data
    data = load(filenames(i).name);
    
    % Add head, K data for successful runs
    numSuccessfulRuns = length(data.modflow_success);
    head_data1_overTime = [head_data1_overTime; data.head_data1(1:numSuccessfulRuns,:)];
    head_data2_overTime = [head_data2_overTime; data.head_data2(1:numSuccessfulRuns,:)];
    
    % Add K, pumping data for successful runs
    hk = [hk; data.hk(1:numSuccessfulRuns)'];
    vka = [vka; data.vka(1:numSuccessfulRuns)'];
    pump_rate1 = [pump_rate1; data.pump_rate_1(1:numSuccessfulRuns)'];
    pump_rate2 = [pump_rate2; data.pump_rate_2(1:numSuccessfulRuns)'];
    
    % Get time and check that consistent with other time vectors
    if i == 1
        time = data.time;
    else
        numTime = length(time);
        sameLength = numTime == length(data.time);
        if ~sameLength
            error('time vectors not the same lenght')
        end
        err = time - data.time;
        tol = 1e-3;
        indexBadMatch = err > tol;
        if sum(indexBadMatch > 0) 
            error('time vectors do not have same values')
        end
    end
end

% Clear data and temp varaibles
clear data err tol indexBadMatch numSuccessfulRuns filenames numFiles
    
%% Aggregate data for time-series neural net model 

% Get number of runs
[numRuns, ~] = size(head_data1_overTime);

outputs = zeros(numRuns * numTime, 2);

% Rehape the output data to have all the data for time period 1, then all
% the data for time period 2, etc
outputs(:,1) = reshape(head_data1_overTime,[numRuns*numTime, 1]);
outputs(:,2) = reshape(head_data2_overTime,[numRuns*numTime, 1]);

inputs = zeros(numRuns * numTime, 5);

% Replicate each static variable so the same series repeats for each time
% period
inputs(:,1) = repmat(hk, [numTime,1]);
inputs(:,2) = repmat(vka, [numTime,1]);
inputs(:,3) = repmat(pump_rate1, [numTime,1]);
inputs(:,4) = repmat(pump_rate2, [numTime,1]);

% Replicate and reshape time to get a vector repeats each time numRuns times, then des
% the same for the next time value
timeTemp = repmat(time, [numRuns, 1]);
inputs(:,5) = reshape(timeTemp, [numRuns*numTime, 1]);





