% Solve an Input-Output Fitting problem with a Neural Network
% Script generated by Neural Fitting app
% Created 16-Aug-2017 08:55:56
%
%% Setup

% Train neural net?
trainNet = true;
% get job ID
jobid = getenv('SLURM_JOB_ID');

%% Combine exisitng .mat files from simulation

% Sampling parameters
runsToUse = 500;
maxDrawdownRuns = 0;
maxTimeRuns = 0; 
sampleTime = false;
maxFileNum = 5;

% Load head data
timeToOpen = '2017-11-09 15:22:35';
headData = [];
runIndex = [];
for i = 0:maxFileNum
    filename = strcat('modflowData_headData',num2str(i), timeToOpen,'.mat');
    data = load(filename);
    headDataTemp = data.headData;
    headData = cat(3, headData, headDataTemp);
    [~,~,runsPerFile] = size(headDataTemp);
    runsThisFile = i*runsPerFile +1:(i+1)*runsPerFile;
    runIndex = [runIndex runsThisFile];
    clear data headDataTemp
end

% Load hk and ss data
filename3 = strcat('modflowData_hk',timeToOpen,'.mat');
filename4 = strcat('modflowData_ss',timeToOpen,'.mat');
data = load(filename3);
hk = data.hk(runIndex); % Make sure get same runs for hk and ss as for headData
clear data
data = load(filename4);
ss = data.ss(runIndex);
clear data

% Load time data
filename6 = strcat('modflowData_nstp',timeToOpen,'.mat');
filename5 = strcat('modflowData_time',timeToOpen,'.mat');
timeData = load(filename5);
timeData = timeData.timeSeries;
nstp = load(filename6);
nstp = nstp.nstp;
numTime = nstp *30;

% Truncate runs: affects both headData and ss, hk
headData = headData(:,:,1:runsToUse);
ss = ss(1:runsToUse);
hk = hk(1:runsToUse);
timeData = timeData(1:runsToUse,:);

% Log transform data
hk = log(hk);
ss = log(ss);
% [a, b, c] = size(headData);
% startingHead = headData(:,1,1);
% startingHead = repmat(startingHead,[1,b,c]);
% drawdown =  startingHead - headData;
% drawdownPlus = drawdown + 100;
% logheadData = log(drawdownPlus);
% temp = logheadData(:,:,1);
% disp('data loaded')


%% Aggregate data for time-series neural net model 
[numWells, ~, numRuns] = size(headData);
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
inputs(:,2) = reshape(repmat(ss(1:numRuns), [numTime,1]),[],1);

% Reshape time to get a vector repeats each time numRuns times, then des
% the same for the next time value
inputs(:,3) = reshape(timeData', [numRuns*numTime, 1]);

disp('inputs and outputs created')

%% GUI-genereated script
x = inputs';
t = outputs';
clear inputs outputs

if trainNet
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest. % Levenberg-Marquardt backpropagation.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  

% Create a Fitting Network
hiddenLayerSize = 6;
%hiddenLayerSize = [4 4];
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Change transfer function
%net.layers{2}.transferFcn = 'purelin';

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Set performance goal
net.trainParam.goal = 30;

% Set number of epochs
net.trainParam.epochs = 1500;

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};
disp('nn setup configured')

% Train the Network
pool = parpool('local', str2num(getenv('SLURM_CPUS_PER_TASK')));
disp('pool started, ready for training')
N = 16;
[net,tr] = train(net,x,t,'CheckpointFile', strcat('MyCheckpoint_', jobid),'useParallel','yes','useGPU','yes','showResources','yes','reduction',N);
    

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

save(strcat('nnoutput_', jobid), 'net')
end

%% Work with network after training and testing

% View the Network
if (false)
    view(net)
end

% Plotsmatlabroot
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
%figure, plottrainstate(tr)
% figure, ploterrhist(e)
% figure, plotregression(t,y)
% figure, plotfit(net,x,t)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (true)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,strcat('myNeuralNetworkFunction_', jobid),'MatrixOnly','yes');
    %y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end
