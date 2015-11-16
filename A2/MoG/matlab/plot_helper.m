data_kmeans = [
       2.8826e+04
   2.9368e+04
   2.8522e+04
   2.8036e+04
   2.8545e+04
   2.9441e+04
   2.7921e+04
   2.9281e+04
   2.9087e+04
   2.9110e+04];
data_random = [
    2.7902e+04
   2.3336e+04
   2.6483e+04
   2.4787e+04
   2.6553e+04
   2.4435e+04
   2.5959e+04
   2.4653e+04
   2.4218e+04
   2.4772e+04];
hold on;
plot(data_kmeans, 'r');
plot(data_random, 'b');
legend('k-means', 'random values', -1);
