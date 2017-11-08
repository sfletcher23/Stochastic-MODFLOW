%% Test neural net performance


%% Neural net script to use
netname = 'myNeuralNetworkFunction_40323';
netscript = str2func(netname); 

%% Create x and t

loadData = true;
runsPerFile = 150;

if loadData == true

    % Load head data
    timeToOpen = '2017-11-08 13:30:38';
    headData = [];
    runIndex = [];
    for i = 8
        filename = strcat('modflowData_headData',num2str(i), timeToOpen,'.mat');
        data = load(filename);
        headDataTemp = data.headData;
        headData = cat(3, headData, headDataTemp);
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
    
    % Time vector
    [numWells, numTime, numRuns] = size(headData);
    time = 1:numTime;

    % Initialize output (targets)
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
    inputs(:,3) = repmat(time', [numRuns, 1]);

    x = inputs';
    t = outputs';
    clear inputs outputs
    
end



%% Plot expected heads vs actual
wellIndex = [55 53];
numSamples = 100;
index = randsample(numRuns, numSamples);
y_estimated = netscript(x(:,index));
y_actual = t(:,index);
figure;
subplot(1,2,1)
scatter(1:numSamples, y_estimated(wellIndex(1),:),'*');
hold on
scatter(1:numSamples, y_actual(wellIndex(1),:));
title('Well 1')
xlabel('sample number')
ylabel('head [m]')
legend('Estimated', 'Actual')
ylim([-600 650])
subplot(1,2,2)
scatter(1:numSamples, y_estimated(wellIndex(2),:), '*');
hold on
scatter(1:numSamples, y_actual(wellIndex(2),:));
title('Well 2')
xlabel('sample number')
ylabel('head [m]')
legend('Estimated', 'Actual')
ylim([-600 650])

%% Plot histogram of errors
numSamples = 100;
index = randsample(numRuns, numSamples);
y_est= netscript(x(:,index));
err = y_est - t(:,index); 
figure
hist(err,100,'k');
h = gca;
h.ColorOrder = repmat([1 1 1], [7, 1]);
set(get(gca,'child'),'FaceColor','k','EdgeColor','k');
xlabel('estimated head - actual head')
ylabel('instances')
title('Error histogran with 100 bins')

%% Test one time series

% Sample run number to plot
i = randsample(numRuns,1);

time = 1:52*30;

% Get row index range of time series corresponding to sample
indexMin = (i-1)*52*30 + 1;
indexMax = i*52*30;

% Get estimates from nn and modflow output
xsample = x(:,indexMin:indexMax);
y_est = netscript(xsample);
y_act = t(:, indexMin:indexMax);

% Plot time series; divide among subplots for readability

for i = 1:9
    figure
    indexMin = (i-1)*12 +1;
    indexMax = i*12;
    set(gca, 'ColorOrder', parula(12), 'NextPlot', 'replacechildren');
    f1 = plot(time,y_est(indexMin:indexMax, :), '-');
    hold on
    set(gca, 'ColorOrder', parula(12));
    f2 = plot(time,y_act(indexMin:indexMax, :), '-.');
%     legend('well 1 est', 'well 2 est', 'well 1 act', 'well 2 act')

end

%% Plot targets vs estimates

numSamples = 250;
index = randsample(numRuns, numSamples);
figure
plot(1:.5:200, 1:.5:200, 'k')
for i = 1:108
    y_est= netscript(x(:,index));
    y_est = y_est(i,:);
    y_act = t(i,index);
    hold on
    scatter(y_act, y_est,1,'*')
end
ylabel('Estimates')
xlabel('Targets')
xlim([150 200])
ylim([150 200])

%% MSE 
y = netscript(x);
mse = sum(sum( (y - t) .^2 )) / numel(y) 


