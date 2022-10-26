
X = out.x(2:end-1,2);
Y = diff(out.x(1:end-1,2));
Z1 = out.F_fric_nl(3:end,2);
Z2 = -40*out.x(3:end,2);

figure;
hold on
scatter3(X,Y,Z1)
scatter3(X,Y,Z2)
xlabel('v')
ylabel('dv')
zlabel('F_R')