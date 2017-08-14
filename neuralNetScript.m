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
time = [];

% Get data as arrays of time
for i = 1:numFiles
    
    % Load data
    data = load(filenames(i).name);
    
    % Add head, K data for successful runs
    numSuccessfulRuns = length(data.modflow_success);
    head_data1_overTime = [head_data1_overTime; data.head_data1(1:numSuccessfulRuns,:)];
    head_data2_overTime = [head_data2_overTime; data.head_data2(1:numSuccessfulRuns,:)];
    
    % Add time data for successful runs
    time = [time; data.timeSeries(1:numSuccessfulRuns,:)];
    
    % Add K, pumping data for successful runs
    hk = [hk; data.hk(1:numSuccessfulRuns)'];
    vka = [vka; data.vka(1:numSuccessfulRuns)'];
    pump_rate1 = [pump_rate1; data.pump_rate_1(1:numSuccessfulRuns)'];
    pump_rate2 = [pump_rate2; data.pump_rate_2(1:numSuccessfulRuns)'];
 
end

% Clear data and temp varaibles
clear data err tol indexBadMatch numSuccessfulRuns filenames numFiles
    
% Aggregate data for time-series neural net model 

% Get number of runs
[numRuns, numTime] = size(head_data1_overTime);

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

% Reshape time to get a vector repeats each time numRuns times, then des
% the same for the next time value
timeTemp = reshape(time,[numRuns*numTime, 1]);
inputs(:,5) = reshape(timeTemp, [numRuns*numTime, 1]);

%% Compare neural net results to modflow results

% Plot expected heads vs actual
numTest = 20;
index = randi([1,numRuns], [numTest,1]);
x = inputs(index,:);
y_estimated = myNeuralNetworkFunction4(x, 1000, 0);
y_actual = outputs(index,:);
figure;
subplot(1,2,1)
scatter(1:numTest, y_estimated(:,1));
hold on
scatter(1:numTest, y_actual(:,1));
title('Well 1')
xlabel('sample number')
ylabel('head [m]')
legend('Estimated', 'Actual')
subplot(1,2,2)
scatter(1:numTest, y_estimated(:,2));
hold on
scatter(1:numTest, y_actual(:,2));
title('Well 2')
xlabel('sample number')
ylabel('head [m]')
legend('Estimated', 'Actual')

% Plot histogram of errors
numTest = 100000;
index = randi([1,numRuns], [numTest,1]);
x = inputs(index,:);
y_estimated = myNeuralNetworkFunction4(x, 1000, 0);
y_actual = outputs(index,:);
err = y_estimated - y_actual; 
figure
hist(err,100,'k');
h = gca
h.ColorOrder = repmat([1 1 1], [7, 1]);
set(get(gca,'child'),'FaceColor','k','EdgeColor','k');
xlabel('estimated head - actual head')
ylabel('instances')
title('Error histogran with 100 bins')

%% Test one time series

filenames = dir('*modflowData*.mat');
numFiles = length(filenames(not([filenames.isdir])));

i = randi([1,numFiles], [1,1]);

% Load data
data = load(filenames(i).name);

% Add head, K data for successful runs
numSuccessfulRuns = length(data.modflow_success);
j = randi([1, numSuccessfulRuns], [1,1]);
head_data1_overTime = data.head_data1(j,:);
head_data2_overTime = data.head_data2(j,:);

% Add time data for successful runs
time = data.timeSeries(j,:);
numTime = length(time);

% Add K, pumping data for successful runs
hk = data.hk(j);
vka = data.vka(j);
pump_rate1 = data.pump_rate_1(j);
pump_rate2 = data.pump_rate_2(j);

% Format x

x = [repmat(hk, [numTime,1]), repmat(vka, [numTime,1]), ...
    repmat(pump_rate1, [numTime,1]), repmat(pump_rate2, [numTime,1]), time'];
y_est = myNeuralNetworkFunction4(x, 1000, 0);
y_act = [head_data1_overTime' head_data2_overTime'];
figure
f1 = plot(time,y_est);
hold on
f2 = plot(time,y_act);
legend('well 1 est', 'well 2 est', 'well 1 act', 'well 2 act')
f1(1).Color = [0 0 1];
f1(2).Color = [0 .5 0.5];
f2(1).Color = [0.12 0.12 0.74];
f2(2).Color = [.2 0.4 0.4];
f1(1).LineStyle = '--'
f1(2).LineStyle = '--'




