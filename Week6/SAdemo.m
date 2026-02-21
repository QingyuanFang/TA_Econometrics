%========================================================================
% Econometrics, Spring 2026
% Prof. Jonathan Wright (wrightj@jhu.edu)
% TA: Qingyuan Fang (qfang6@jhu.edu)

% TA Session, Feb 23rd 2026

% Contents:
%   1) Rosenbrock: visualize path + objective/temperature traces
%   2) Periodic nonlinear least squares: multi-modal objective in frequency
%   3) Compare with MATLAB's simulannealbnd (if available)

%========================================================================

restoredefaultpath
close all;
clear;
clc;
rng(666);

%% 1) Rosenbrock
fprintf('\n[1] Rosenbrock demo...\n');

rosen = @(x) (1 - x(1)).^2 + 100*(x(2) - x(1).^2).^2;

    %% 3-D surface of Rosenbrock for visualization
    [x1g,x2g] = meshgrid(linspace(-2.0,2.0,200), linspace(-1,3,200));
    zg = (1 - x1g).^2 + 100*(x2g - x1g.^2).^2;
    figure('Name','Rosenbrock: 3D surface');
    surf(x1g, x2g, zg, 'EdgeColor','none'); hold on;
    colormap(jet); colorbar;
    shading interp;
    view(45,30);
    xlabel('x_1'); ylabel('x_2'); zlabel('f(x)');
    title('Rosenbrock function surface');
    % Mark global minimum
    plot3(1,1,0,'kp','MarkerSize',12,'MarkerFaceColor','k');

    %% Optimization
    x0 = [-2; 2];
    lb = [-3; -3];
    ub = [ 3;  3];

    opts = sa_defaults();
    opts.T0           = 2.0;     % try 0.5 to see "too cold"
    opts.alpha        = 0.95;    % cooling rate
    opts.nPerTemp     = 200;     % proposals per temperature level
    opts.maxTempIters = 80;      % number of temperature levels
    opts.step0        = [0.4; 0.4];
    opts.adaptSteps   = true;

    [xbest, fbest, hist] = sa_minimize(rosen, x0, lb, ub, opts);

    fprintf('Rosenbrock best f = %.4g at x = [%.4f, %.4f]\n', fbest, xbest(1), xbest(2));

    %% Plot: Contour + path
    figure('Name','Rosenbrock: contour + SA path');
    [X1,X2] = meshgrid(linspace(-2.5,2.5,250), linspace(-1,3,250));
    Z = (1 - X1).^2 + 100*(X2 - X1.^2).^2;
    contour(X1, X2, log10(Z+1e-6), 30); hold on;

    % color the accepted path by temperature: map T to colormap
    Ts = hist.T; % temperature per iteration (same length as xAccepted)
    X = hist.xAccepted;
    npts = size(X,2);

    Tmin = min(Ts);
    Tmax = max(Ts);
    Tnorm = (Ts - Tmin) ./ max(eps, (Tmax - Tmin));
    cmap = jet(256);
    ci = max(1, round(1 + Tnorm*(size(cmap,1)-1)));

    for k = 1:(npts-1)
        plot(X(1,k:k+1), X(2,k:k+1), '-', 'Color', cmap(ci(k),:), 'LineWidth', 0.5); hold on;
    end
    scatter(X(1,:), X(2,:), 3, cmap(ci,:), 'filled');

    plot(1,1,'kp','MarkerSize',12,'MarkerFaceColor','k');
    xlabel('x_1'); ylabel('x_2');
    title('Rosenbrock: SA accepted path (log10 contour)'); grid on;

    %% Plot Traces
    figure('Name','Rosenbrock: objective + temperature traces');
    subplot(2,1,1);
    semilogy(hist.fCurrent, '-'); hold on;
    semilogy(hist.fBestSoFar, '-');
    legend('current f','best-so-far f','Location','best'); grid on;
    xlabel('iteration'); ylabel('objective');
    title('Objective trace');

    subplot(2,1,2);
    plot(hist.T, '-'); grid on;
    xlabel('iteration'); ylabel('temperature');
    title('Temperature trace');

    fprintf('Acceptance rate (overall): %.2f\n', mean(hist.accepted));

%% 2) Periodic NLS
fprintf('\n[2] Periodic NLS (multi-modal in frequency) demo...\n');

% simulate the data
T = 80;
t = (1:T)';
alpha_true = 0.7;
beta_true  = 1;
omega_true = 0.35*pi;  
y = alpha_true + beta_true*sin(omega_true*t) + 0.5*randn(T,1);

% Parameters: theta = [alpha; beta; omega]
sse = @(theta) sum( (y - (theta(1) + theta(2)*sin(theta(3)*t))).^2 );

% Bounds: omega in [0, pi]
lb2 = [-5; -5; 0   ];
ub2 = [ 5;  5; pi];

% Bad local start on purpose
theta0 = [0; 0.5; 0.9*pi];
theta_true = [alpha_true;beta_true;omega_true];

% Local baseline: fminsearch with a bounded transform
fprintf('Running local baseline (fminsearch with transform)...\n');
theta_local = local_fminsearch_bounded(sse, theta0, lb2, ub2);
fprintf('Local baseline SSE = %.4g, omega=%.4f (true %.4f)\n', sse(theta_local), theta_local(3), omega_true);

% SA run
opts2 = sa_defaults();
opts2.T0           = 10.0;
opts2.alpha        = 0.97;
opts2.nPerTemp     = 300;
opts2.maxTempIters = 120;
opts2.step0        = [0.2; 0.3; 0.25];
opts2.adaptSteps   = true;

fprintf('Running SA...\n');
[theta_sa, sse_sa, hist2] = sa_minimize(sse, theta0, lb2, ub2, opts2);

fprintf('SA SSE = %.4g, omega=%.4f (true %.4f)\n', sse_sa, theta_sa(3), omega_true);

% Local polish from SA output
fprintf('Local polish from SA output...\n');
theta_polish = local_fminsearch_bounded(sse, theta_sa, lb2, ub2);
fprintf('Polished SSE = %.4g, omega=%.4f (true %.4f)\n', sse(theta_polish), theta_polish(3), omega_true);

% Trace for SA in periodic NLS
figure('Name','Periodic NLS: SA trace');
subplot(2,1,1);
semilogy(hist2.fCurrent, '-'); hold on;
semilogy(hist2.fBestSoFar, '-'); grid on;
axis tight
legend('current SSE','best-so-far SSE','Location','best');
xlabel('iteration'); ylabel('SSE');

subplot(2,1,2);
plot(hist2.xCurrent(3,:), '-'); hold on; grid on;
yline(omega_true, '--');
axis tight
xlabel('iteration'); ylabel('\omega');
title('Frequency coordinate over iterations');

%% 3) MATLAB toolbox SA
if exist('simulannealbnd','file') == 2
    fprintf('\n[3] Optional: comparing with simulannealbnd (toolbox detected)...\n');
    try
        saopts = optimoptions('simulannealbnd', ...
            'MaxIterations', 100000, ...
            'Display','final');
        [th_mw, f_mw] = simulannealbnd(sse, theta0, lb2, ub2, saopts);
        fprintf('simulannealbnd SSE = %.4g, omega=%.4f\n', f_mw, th_mw(3));
    catch ME
        fprintf('simulannealbnd call failed: %s\n', ME.message);
    end
end

if exist('th_mw','var')
    theta_MATLAB = th_mw(:);
    sse_MATLAB = sse(theta_MATLAB);
else
    % if MATLAB toolbox run didn't occur, fill with NaNs
    theta_MATLAB = nan(3,1);
    sse_MATLAB = NaN;
end

col1 = [theta_true; sse(theta_true)];
colf = [theta_local; sse(theta_local)];
col2 = [theta_sa;   sse_sa];
col3 = [theta_polish; sse(theta_polish)];
col4 = [theta_MATLAB; sse_MATLAB];

Ttbl = table(col1, colf, col2, col3, col4, 'RowNames', {'alpha','beta','omega','SSE'});
Ttbl.Properties.VariableNames = {'True','fminsearch','SA','PolishedFromSA','MATLAB_toolbox'};
disp('Parameter estimates and SSEs:');
disp(Ttbl);

%% Helpers
function opts = sa_defaults()
    opts.T0           = 1.0;     % initial temperature
    opts.Tmin         = 1e-6;    % minimum temperature
    opts.alpha        = 0.95;    % T <- alpha*T
    opts.nPerTemp     = 200;     % proposals per temperature level
    opts.maxTempIters = 100;     % number of temperature levels
    opts.step0        = 0.2;     % scalar or vector step size
    
    % for adjusting step size
    opts.adaptSteps   = true;    % adapt step sizes using acceptance rate
    opts.targetAccLo  = 0.2;
    opts.targetAccHi  = 0.6;
    opts.stepGrow     = 1.15;
    opts.stepShrink   = 0.75;

    opts.stallTol     = 1e-10;   % improvement threshold
    opts.stallIters   = 10;      % number of temp-levels with no improvement before stopping
    opts.verbose      = true;
end

function [xbest, fbest, hist] = sa_minimize(fun, x0, lb, ub, opts)
% SA_MINIMIZE  Bounded simulated annealing (reflection at bounds).
% Inputs:
%   fun : objective handle, fun(x) -> scalar
%   x0  : initial point (column vector)
%   lb, ub : bounds (same size as x0)
%   opts : struct (see sa_defaults)
%
% Outputs:
%   xbest, fbest : best solution found
%   hist : struct of traces

    x = x0(:);
    d = numel(x);
    lb = lb(:); ub = ub(:);

    if isscalar(opts.step0)
        step = opts.step0 * ones(d,1);
    else
        step = opts.step0(:);
    end

    f = safe_eval(fun, x);
    xbest = x; fbest = f;

    maxIter = opts.maxTempIters * opts.nPerTemp;

    hist.xCurrent    = nan(d, maxIter);
    hist.xAccepted   = nan(d, maxIter);
    hist.fCurrent    = nan(1, maxIter);
    hist.fBestSoFar  = nan(1, maxIter);
    hist.T           = nan(1, maxIter);
    hist.accepted    = false(1, maxIter);

    T = opts.T0;
    iter = 0;
    acc_this_temp = 0;
    prop_this_temp = 0;

    stall_count = 0;
    best_before_temp = fbest;
    
    % Outer Loop for temperature
    for k = 1:opts.maxTempIters
        % Inner Loop for proposal
        for j = 1:opts.nPerTemp
            iter = iter + 1;

            % propose
            xnew = x + step .* randn(d,1);
            xnew = reflect_bounds(xnew, lb, ub);

            fnew = safe_eval(fun, xnew);
            df = fnew - f;

            % accept?
            accept = false;
            if df <= 0
                accept = true;
            else
                % prevent overflow/underflow issues
                p = exp(-df / max(T, realmin));
                if rand() < p
                    accept = true;
                end
            end

            if accept
                x = xnew; f = fnew;
                acc_this_temp = acc_this_temp + 1;
                hist.xAccepted(:,iter) = x;
            else
                % hist.xAccepted(:,iter) = nan(d,1);
                hist.xAccepted(:,iter) = x;
            end

            prop_this_temp = prop_this_temp + 1;

            if f < fbest
                fbest = f; xbest = x;
            end

            % record traces
            hist.xCurrent(:,iter)   = x;
            hist.fCurrent(iter)     = f;
            hist.fBestSoFar(iter)   = fbest;
            hist.T(iter)            = T;
            hist.accepted(iter)     = accept;
        end

        % End of temperature level: adapt step sizes based on acceptance rate
        acc_rate = acc_this_temp / max(prop_this_temp,1);
        if opts.adaptSteps
            if acc_rate > opts.targetAccHi
                step = step * opts.stepGrow;
            elseif acc_rate < opts.targetAccLo
                step = step * opts.stepShrink;
            end
        end

        % stall detection (best-so-far improvements across temperature levels)
        if (best_before_temp - fbest) <= opts.stallTol
            stall_count = stall_count + 1;
        else
            stall_count = 0;
        end
        best_before_temp = fbest;

        if opts.verbose
            fprintf('Temp-level %3d | T=%9.3g | acc=%.2f | fbest=%12.6g | step~%.3g\n', ...
                k, T, acc_rate, fbest, mean(step));
        end
        
        % stop criteria
        if T < opts.Tmin || stall_count >= opts.stallIters
            break;
        end

        % cool
        T = opts.alpha * T;

        % reset counters for next temperature level
        acc_this_temp = 0;
        prop_this_temp = 0;
    end

    % trim history to actual iterations
    hist.xCurrent    = hist.xCurrent(:,1:iter);
    hist.xAccepted   = hist.xAccepted(:,1:iter);
    hist.fCurrent    = hist.fCurrent(1:iter);
    hist.fBestSoFar  = hist.fBestSoFar(1:iter);
    hist.T           = hist.T(1:iter);
    hist.accepted    = hist.accepted(1:iter);
end

function fx = safe_eval(fun, x)
    fx = fun(x);
    if ~isfinite(fx)
        fx = inf;
    end
end

function x = reflect_bounds(x, lb, ub)
% Reflect component-wise at bounds until inside [lb,ub]
    for i = 1:numel(x)
        if x(i) < lb(i)
            x(i) = 2*lb(i) - x(i);
        elseif x(i) > ub(i)
            x(i) = 2*ub(i) - x(i);
        end
        % if reflection still out (huge step), just clip
        if x(i) < lb(i), x(i) = lb(i); end
        if x(i) > ub(i), x(i) = ub(i); end
    end
end

function xhat = local_fminsearch_bounded(fun, x0, lb, ub)
% Use a smooth transform to unconstrained space for fminsearch:
%   z in R^d -> x = lb + (ub-lb) .* sigmoid(z)
% With inverse:
%   z = log( (x-lb)/(ub-x) )

    x0 = x0(:); lb = lb(:); ub = ub(:);
    z0 = inv_sigmoid((x0 - lb) ./ (ub - lb));
    obj_z = @(z) fun(lb + (ub-lb).*sigmoid(z));

    zhat = fminsearch(obj_z, z0, optimset('Display','off'));
    xhat = lb + (ub-lb).*sigmoid(zhat);
end

function s = sigmoid(z)
    s = 1 ./ (1 + exp(-z));
end

function z = inv_sigmoid(s)
    s = min(max(s, 1e-12), 1-1e-12);
    z = log(s ./ (1 - s));
end


