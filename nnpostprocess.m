

%% Compare neural net results to modflow results

% Plot expected heads vs actual
numTest = 100;
index = randi([1,numRuns], [numTest,1]);
% x = inputs(index,:);
y_estimated = myNeuralNetworkFunction(x(:,index));
y_actual = y(:,index);
figure;
subplot(1,2,1)
scatter(1:numTest, y_estimated(108,:),'*');
hold on
scatter(1:numTest, y_actual(108,:));
title('Well 1')
xlabel('sample number')
ylabel('head [m]')
legend('Estimated', 'Actual')
ylim([0 200])
subplot(1,2,2)
scatter(1:numTest, y_estimated(66,:), '*');
hold on
scatter(1:numTest, y_actual(66,:));
title('Well 2')
xlabel('sample number')
ylabel('head [m]')
legend('Estimated', 'Actual')
ylim([0 200])

% Plot histogram of errors
numTest = 10000;
index = randi([1,numRuns], [numTest,1]);
y_est= myNeuralNetworkFunction(x(:,index));
err = y_est - t(:,index); 
figure
hist(err,100,'k');
h = gca
h.ColorOrder = repmat([1 1 1], [7, 1]);
set(get(gca,'child'),'FaceColor','k','EdgeColor','k');
xlabel('estimated head - actual head')
ylabel('instances')
title('Error histogran with 100 bins')

%% Test one time series

% Sample run number to plot
i = randi([1 numRuns]);

% Get row index range of time series corresponding to sample
indexMin = (i-1)*365*30 + 1;
indexMax = i*365*30;

% Get estimates from nn and modflow output
xsample = x(:,indexMin:indexMax);
y_est = myNeuralNetworkFunction(xsample);
y_act = t(:, indexMin:indexMax);

% Plot time series; divide among subplots for readability

for i = 1:9
    figure
    indexMin = (i-1)*12 +1;
    indexMax = i*12;
    set(gca, 'ColorOrder', parula(12), 'NextPlot', 'replacechildren');
    f1 = plot(time,y_est(indexMin:indexMax,:), '-');
    hold on
    set(gca, 'ColorOrder', parula(12));
    f2 = plot(time,y_act(indexMin:indexMax,:), '-.');
%     legend('well 1 est', 'well 2 est', 'well 1 act', 'well 2 act')

end

%% Save results
save(strcat('nnresults', timeToOpen))


