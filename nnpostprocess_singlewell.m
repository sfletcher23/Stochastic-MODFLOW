%% Test neural net performance


%% Neural net script to use
netname = 'myNeuralNetworkFunction_17457';
netscript = str2func(netname); 
wellIndex = 108;

%% Create x and t

loadData = true;

if loadData == true

    % Load head data
    timeToOpen = '2017-08-15 14:31:17';
    headData = [];
    runIndex = [];
    for i = 11
        filename = strcat('modflowData_headData',num2str(i), timeToOpen,'.mat');
        data = load(filename);
        headDataTemp = data.headData;
        headData = cat(3, headData, headDataTemp(wellIndex, :, :));
        runsThisFile = i*250 +1:(i+1)*250;
        runIndex = [runIndex runsThisFile];
        clear data headDataTemp
    end

    % Load hk and sy data
    filename3 = strcat('modflowData_hk',timeToOpen,'.mat');
    filename4 = strcat('modflowData_sy',timeToOpen,'.mat');
    data = load(filename3);
    hk = data.hk(runIndex); % Make sure get same runs for hk and sy as for headData
    clear data
    data = load(filename4);
    sy = data.sy(runIndex);
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
    inputs(:,2) = reshape(repmat(sy(1:numRuns), [numTime,1]),[],1);

    % Reshape time to get a vector repeats each time numRuns times, then des
    % the same for the next time value
    inputs(:,3) = repmat(time', [numRuns, 1]);

    x = inputs';
    t = outputs';
    clear inputs outputs
    
end



%% Plot expected heads vs actual
% wellIndex = randsample(108, 2);
numSamples = 100;
index = randsample(numRuns, numSamples);
y_estimated = netscript(x(:,index));
y_actual = t(:,index);
figure;
scatter(1:numSamples, y_estimated,'*');
hold on
scatter(1:numSamples, y_actual);
title('Well 1')
xlabel('sample number')
ylabel('head [m]')
legend('Estimated', 'Actual')
ylim([0 200])

%% Plot histogram of errors
numSamples = 1000;
index = randsample(numRuns*numTime, numSamples);
y_est= netscript(x(:,index));
t_sample = t(:,index);

% Include only those above the depth limit
indexAboveLimit = t_sample > 100;
t_sample = t_sample(indexAboveLimit);
y_est = y_est(indexAboveLimit);
err = y_est - t_sample; 
figure
hist(err,100,'k');
h = gca;
h.ColorOrder = repmat([1 1 1], [7, 1]);
set(get(gca,'child'),'FaceColor','k','EdgeColor','k');
xlabel('Estimated head (ANN) - Target head (MODFLOW)')
ylabel('Instances')
title('Error Histogram')
xlim([-40 80])
xticks(-40:10:80)


%% Test one time series


time = 1:365*30;

figure;
% Plot time series; divide among subplots for readability
for k = 1:20
    i = randsample(numRuns,1);
    % Get row index range of time series corresponding to sample
    indexMin = (i-1)*365*30 + 1;
    indexMax = i*365*30;

    % Get estimates from nn and modflow output
    xsample = x(:,indexMin:indexMax);
    y_est = netscript(xsample);
    y_act = t(:, indexMin:indexMax);

    hold on
    set(gca, 'ColorOrder', parula(12), 'NextPlot', 'replacechildren');
    f1 = plot(time,y_est, '-');
    set(gca, 'ColorOrder', parula(12));
    hold on
    f2 = plot(time,y_act, '-.');
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
rmse = sqrt(mse)

indexAboveLimit = t > 100;
mse_aboveLimit = sum(sum( (y(indexAboveLimit) - t(indexAboveLimit)) .^2 )) / numel(y(indexAboveLimit))
rmse_aboveLimit = sqrt(mse_aboveLimit)


