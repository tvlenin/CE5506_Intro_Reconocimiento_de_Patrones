
u   =    [0.009, 0.01, 0.008];
gamma =  [0.11, 0.1, 0.12];
training=[69, 70, 65];
new_data=[15, 14.1, 15];

figure(1);
plot3(u,gamma,training);

hold off;

figure(2);
plot3(u,gamma,new_data);