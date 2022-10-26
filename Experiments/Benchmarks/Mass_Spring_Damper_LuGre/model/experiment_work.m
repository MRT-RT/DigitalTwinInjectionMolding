%%
%close all
MassSpringDamper_nonlinear_init;

%% Conduct experiment for training data

% Design input signal
[u,F,f,f_exc] = RandomPhaseMultisineRandomGrid(fs,df,fband,Nsub,Nrep,mode,energy);
% Calculate frequencies where odd and even nonlinearities will appear
[f_even,f_odd] = calculate_nonlinearity_bins(fs,df,f_exc,mode);


%% Simulate the system
simout = sim('MassSpringDamper_nonlinear', T_sim);

[~,ind] = ismembertol([0:T_s:T_sim],simout.yout{1}.Values.time);
y = getdatasamples(simout.yout{1}.Values,ind);
y = timeseries(y,[0:T_s:T_sim]);

%%
% load('dataset1.mat')
% u=timeseries(dataset1(:,1),[0:T_s:T_sim]);
% y=timeseries(dataset1(:,2),[0:T_s:T_sim]);

%% Get rid of transient by deleting observations from first few periods
[u2,y2,T2] = GetRidofTransient(u,y,0,fs,df);

%% Calculate Input and Output Spectrum
f2 = [0 : fs/length(y2.Data): fs/2];            % Recalculate frequency bins
[~,f_exc_bin] = ismembertol(f_exc,f2);          % Recalulate excited frequency bins
[~,f_even_bin] = ismembertol(f_even,f2);        % Recalulate even nonlinearity frequency bins
[~,f_odd_bin] = ismembertol(f_odd,f2);          % Recalulate odd nonlinearity frequency bins

U2 = fft(u2.Data)';
Y2 = fft(y2.Data)';

U2_amp = 2 * abs(U2(1:length(u2.Data)/2+1));
Y2_amp = 2 * abs(Y2(1:length(y2.Data)/2+1));

figure;
subplot(3,1,1);
hold on
plot(f2,db(U2_amp),'x')
plot(f2(f_exc_bin),db(U2_amp(f_exc_bin)),'xk') % plot excited frequencies in black
hold off
%xlim([0,2*fband(2)])

subplot(3,1,2);
hold on
plot(f2,db(Y2_amp),'x')
plot(f2(f_exc_bin),db(Y2_amp(f_exc_bin)),'xk') % plot excited frequencies in black
plot(f2(f_even_bin),db(Y2_amp(f_even_bin)),'xg') % plot excited frequencies in green
plot(f2(f_odd_bin),db(Y2_amp(f_odd_bin)),'xr') % plot excited frequencies in red
hold off
%xlim([0,2*fband(2)])

subplot(3,1,3);
hold on
plot(f2(f_exc_bin),db(Y2_amp(f_exc_bin)./U2_amp(f_exc_bin)),'x')
% plot(f2(f_exc_bin),db(Y2_amp(f_exc_bin)),'xk') % plot excited frequencies in black
% plot(f2(f_even_bin),db(Y2_amp(f_even_bin)),'xg') % plot excited frequencies in green
% plot(f2(f_odd_bin),db(Y2_amp(f_odd_bin)),'xr') % plot excited frequencies in red
hold off
%xlim([0,2*fband(2)])

dataset1 = [u.data,y.data];
save('dataset1.mat','dataset1')

%% Plot
close all

figure;
scatter(simout.x(:,2),-simout.F_fric_nl(:,2)+simout.x(:,2).*d/m)
xlabel('$\dot{x}$','fontsize',14,'interpreter','latex')
ylabel('$F_{R}(\dot{x})$','fontsize',14,'interpreter','latex')
saveas(gcf,'FrictionForce.png')

figure;
scatter3(simout.x(:,2),simout.z,-simout.F_fric_nl(:,2)+simout.x(:,2).*d/m)
xlim([-0.04,0.04])
xlabel('$\dot{x}$','fontsize',14,'interpreter','latex')
ylabel('$z$','fontsize',14,'interpreter','latex')
saveas(gcf,'FrictionForce3d.png')

figure;
hist(simout.x(:,2))
xlabel('$\dot{x}$','fontsize',14,'interpreter','latex')
saveas(gcf,'HistSpeed.png')

%% Conduct experiment for validation data
% Design input signal
[u,F,f,f_exc] = RandomPhaseMultisineRandomGrid(fs,df,fband,Nsub,Nrep,mode,energy);

simout = sim('MassSpringDamper_nonlinear', T_sim);

[~,ind] = ismembertol([0:T_s:T_sim],simout.yout{1}.Values.time);
y = getdatasamples(simout.yout{1}.Values,ind);
y = timeseries(y,[0:T_s:T_sim]);

dataset2 = [u.data,y.data];

save('dataset2.mat','dataset2')


%% Conduct experiment for test data
% Design input signal
[u,F,f,f_exc] = RandomPhaseMultisineRandomGrid(fs,df,fband,Nsub,Nrep,mode,energy);

simout = sim('MassSpringDamper_nonlinear', T_sim);

[~,ind] = ismembertol([0:T_s:T_sim],simout.yout{1}.Values.time);
y = getdatasamples(simout.yout{1}.Values,ind);
y = timeseries(y,[0:T_s:T_sim]);

dataset3 = [u.data,y.data];

save('dataset3.mat','dataset3')
