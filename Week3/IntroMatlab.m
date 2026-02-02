%========================================================================
% Econometrics, Spring 2026
% Prof. Jonathan Wright (wrightj@jhu.edu)
% TA: Qingyuan Fang (qfang6@jhu.edu)

% TA Session, Feb 2nd 2026
%========================================================================

restoredefaultpath
close all;          % closes all figures
clear;              % removes variables from workspace
clc;                % clears command window
rng(666, "twister");% set random seed for reproducible simulations


% =========================================================================
% 1) Sections start with %%
%    Run section-by-section with: Ctrl+Enter (Windows) / Cmd+Enter (Mac).
%    or: Click the blue bar
% 2) Inspect variables in Workspace, or use "whos".
% 3) Help / documentation:
%    help functionName
%    doc functionName
% Example:
% help inv
% doc mldivide
%
% Big MATLAB facts to remember:
%   - 1-indexed (first element is x(1), not x(0))
%   - Column vectors are the default in math/econometrics (T-by-1)
%   - Matrix operations matter: * / \ are linear algebra, .* ./ .^ are elementwise
%   - Preallocate arrays; avoid growing in loops when T is large
% =========================================================================

%% 0. Display format   

format short;     % readable numeric display
disp(pi)

format long;
disp(pi)

format short E;
disp(pi)

format short g;
disp(pi)

fprintf("This is a pie: %.3f \n",pi)

%% 1. Scalars, vectors, matrices, and basic operations
a = 2;              % scalar
b = 3.5;            % scalar (double precision by default)
c = a + b;

% Semicolons suppress output (useful to keep the command window clean)
d = a*b;

% Row vector vs column vector
x_row = [1, 2, 3, 4]; % comma
x_col = [1; 2; 3; 4]; % colon

% Common constructors:
z = zeros(3,2);     % 3x2 zeros
o = ones(2,4);      % 2x4 ones
oo = ones(3);       % 3x3 ones
I = eye(4);         % 4x4 identity
A = rand(3,3);      % uniform(0,1) random
B = randn(3,3);     % standard normal random

% Size and shape
[nr, nc] = size(B);
L = length(x_row);  % length = max(size(...)) (can be misleading)
N = numel(B);       % number of elements (always safe)

whos a b c x_row x_col A B

%% 2. Indexing (1-based) and slicing
v = (10:2:20);       % 10,12,14,16,18,20 (note: start and end inclusive)
w = linspace(0,1,6); % 6 evenly spaced points between 0 and 1

% Indexing vectors
v1 = v(1);           % first element
v_end = v(end);      % last element
v_mid = v(2:4);      % elements 2 to 4

% Matrices and slicing
M = reshape(1:12, 3, 4);  % fills column-wise by default
% M =
%   1  4  7  10
%   2  5  8  11
%   3  6  9  12

M_23 = M(2,3);        % row 2, col 3
M_row2 = M(2,:);      % entire second row
M_col3 = M(:,3);      % entire third column
M_block = M(1:2,2:4); % submatrix

% Logical indexing (econometrics workhorse)
idx = (M(:,1) >= 2);  % logical vector: which rows satisfy condition?
M_rows = M(idx,:);

%% 3. Elementwise vs matrix operations (critical)
X = [1 2; 
     3 4];
Y = [10 20; 
     30 40];

% Matrix product
XY = X * Y;

% Elementwise operations
X_times_Y = X .* Y;
X_div_Y   = X ./ Y;
X_pow2    = X .^ 2;

% Transpose:
%   '  is conjugate transpose (same as transpose for real numbers)
%   .' is transpose without conjugation (useful for complex arrays)
Xt = X';   
Xtt = X.';

% A = [1+2i, 3+4i; 
%      5+6i, 7+8i];
% 
% % Conjugate transpose (A')
% B = A';
% % B will be:
% %   1-2i   5-6i
% %   3-4i   7-8i
% 
% % Nonconjugate transpose (A.')
% C = A.';
% % C will be:
% %   1+2i   5+6i
% %   3+4i   7+8i

%% 4. Broadcasting / implicit expansion (modern MATLAB)
% You often want to subtract a mean vector from each row/column.
T = 5; 
K = 3;

Data = randn(T,K);
mu = mean(Data,1);       % 1-by-K mean
Data_demean = Data - mu; % subtract mu from each row (implicit expansion)

%% 5. Assertions, sanity checks, numerical tolerance
% Floating-point arithmetic: never compare doubles with == unless you know why.
u = 0.1 + 0.2;
v = 0.3;
diff_uv = abs(u - v);

tol = 1e-12;
assert(diff_uv < tol, 'Unexpected floating point discrepancy.');

% Sanity checks on shapes
assert(size(Data,1) == T && size(Data,2) == K, 'Data has unexpected shape.');

%% 6. Plotting basics
x = linspace(0,2*pi,400)';
y1 = sin(x);
y2 = cos(x);

figure;
plot(x, y1, 'LineWidth', 1.5); hold on;
plot(x, y2, 'LineWidth', 1.5);
xlabel('x'); ylabel('value');
title('Sine and Cosine');
legend({'sin(x)','cos(x)'}, 'Location','best');
grid on;
axis tight;

% Save figure (vector PDF is great for LaTeX)
% exportgraphics(gcf, 'trig_plot.pdf', 'ContentType','vector');

%% 7. Cells, structs, and tables
% Cell arrays: can hold mixed types (strings, matrices, etc.)
C = cell(3,1);
C{1} = 'SPF';
C{2} = randn(2,2);
C{3} = (1:5)';

% Struct: named fields (useful for parameters)
param = struct();
param.beta  = 0.99;
param.rho   = 0.9;
param.sigma = 0.02;

% Table: modern way to handle datasets with variable names
Tobs = 10;
Year = (2000:2000+Tobs-1)';
Infl = 2 + 0.5*randn(Tobs,1);
Unemp = 5 + 0.8*randn(Tobs,1);

tbl = table(Year, Infl, Unemp);
disp(tbl(1:5,:));

% Access table variables
infl_vec = tbl.Infl;
tbl.RealRate = 1.5 + 0.2*randn(Tobs,1);  % add new variable

%% 8. File I/O: saving/loading MAT files and CSV 
% Save workspace objects (MAT format preserves types)
save('demo_data.mat', 'tbl', 'param');

clear tbl param
load('demo_data.mat', 'tbl', 'param');
disp(param);

% CSV is common but loses some metadata (e.g., types)
writetable(tbl, 'demo_data.csv');
tbl2 = readtable('demo_data.csv');

%% 9. Linear algebra
rng(7);

nA = 5;
A = randn(nA);
b = randn(nA,1);

% Avoid inv(A)*b. Use backslash: A\b (numerically stable).
x1 = A \ b;

% Check residual
res = norm(A*x1 - b);
fprintf('Residual norm ||Ax-b|| = %.2e\n', res);

% Condition number matters
condA = cond(A);
fprintf('cond(A) = %.2e (bigger means more ill-conditioned)\n', condA);

% % QR decomposition (used in stable OLS)
% [Q,R] = qr(A);
% x_qr = R \ (Q' * b);
% 
% fprintf('Difference between \\ and QR solution: %.2e\n', norm(x1 - x_qr));

%% 10. OLS from scratch
% Model: y = X*beta + u
% We simulate data with known beta, then estimate it.

rng(202);
T = 500;
K = 3;

X = [ones(T,1), randn(T,K-1)]; % include an intercept
beta_true = [1.0; -0.5; 0.8];
u = 0.5 * randn(T,1);
y = X*beta_true + u;

% OLS estimator: beta_hat = (X'X)^{-1} X'y
beta_hat = (X' * X) \ (X' * y);

% Residuals and sigma^2
uhat = y - X*beta_hat;
s2 = (uhat' * uhat) / (T - K);

% Homoskedastic variance: s2 * (X'X)^{-1}
V_classic = s2 * inv(X' * X);            
se_classic = sqrt(diag(V_classic));

% Robust (HC1) variance:
% V = (X'X)^{-1} (X' diag(uhat.^2) X) (X'X)^{-1} * T/(T-K)
XtX_inv = inv(X' * X);
S = zeros(K,K);
for t = 1:T
    xt = X(t,:)';
    S = S + (uhat(t)^2) * (xt * xt');
end

V_HC0 = XtX_inv * S * XtX_inv;
V_HC1 = (T/(T-K)) * V_HC0;
se_HC1 = sqrt(diag(V_HC1));

% Summarize
tstat_classic = beta_hat ./ se_classic;
tstat_HC1     = beta_hat ./ se_HC1;

disp('True beta vs estimated beta:');
disp(table(beta_true, beta_hat, se_classic, se_HC1, tstat_classic, tstat_HC1, ...
    'VariableNames', {'beta_true','beta_hat','se_classic','se_HC1','t_classic','t_HC1'}));