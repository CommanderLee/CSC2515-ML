data = [    0.2500  100.0000   12.7873    0.0150   16.5649    0.1440
    0.2500  300.0000    4.5798         0   14.1686    0.1200
    0.2500  500.0000    2.6538         0    9.8918    0.0840
    0.5000  100.0000    5.9369         0   12.8735    0.1040
    0.5000  300.0000    2.2785         0   13.7220    0.1080
    0.5000  500.0000    1.3987         0   12.7365    0.1120
    1.0000  100.0000    2.8097         0   15.0511    0.1160
    1.0000  300.0000    1.0870         0   14.5398    0.1080
    1.0000  500.0000    0.6832         0   13.7407    0.1040];

% figure(multi_times+1);
x = 1:9;
subplot(2,1,1);
plot(x, data(:,3), 'r-o', x, data(:,5), 'b:*');
xlabel('hyperparameter sets id');
ylabel('cross entropy');
legend('train','validation', -1);
% title('mnist\_train');
subplot(2,1,2);
plot(x, data(:,4), 'r-o', x, data(:,6), 'b:*');
xlabel('hyperparameter sets id');
ylabel('classification error');
legend('train','validation', -1);
% title('mnist\_train');