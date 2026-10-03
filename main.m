% --------------------------------------------------------------- 
% MBGD MLP (tanh hidden, linear output)
% - Constant LR, mini-batch
% - Per-epoch TRAIN & VAL logging (MSE/RMSE), gradient check
% ---------------------------------------------------------------
close all; rng('default');

%% ---------- Hyperparams ----------
H_list     = [5 10 50 100 150];   % hidden widths to sweep
max_epochs = 100;            
eta        = 5e-4;           
batch_size = 128;

% Radial Fourier features (for Qs3(k) only)
%K_radial   = 4;      

%student_id = '2021315234';

%% ---------- ID-derived params ----------
P = params_from_student_id(student_id);
fprintf('ID=%s | seed=%u | alpha=%.6f | c=%.6f | sigma=%.6f | B=%d\n', ...
    student_id, P.seed, P.alpha_u, P.c_u, P.sigma, batch_size);
rng(P.seed);


%% ------------------ Data (grid) ------------------
nPerAxis = 500; xmin=-2; xmax=2; ymin=-2; ymax=2;
xs = linspace(xmin,xmax,nPerAxis); ys = linspace(ymin,ymax,nPerAxis);
[XX,YY] = meshgrid(xs,ys);
X = [XX(:), YY(:)];                    % [N x 2]

% Target Function
f = @(x,y) -(1+cos(12*P.alpha_u*sqrt(x.^2 + y.^2))) ./ ...
                 (P.alpha_u*(x.^2 + y.^2)+(2+P.c_u));
Z_clean = f(X(:,1), X(:,2));

% Repro checksum surrogate
raw = [X Z_clean]; raw_bytes = typecast(reshape(raw.',1,[]),'uint8'); %#ok<TYPECAST>
fprintf('Checksum = %.10f\n', sum(double(raw_bytes)));

% Split 70/30
N  = size(X,1); idx = randperm(N);
Ntr = max(200, round(0.7*N));
tr = idx(1:Ntr); va = idx(Ntr+1:end);

Xtr = X(tr,:);  Xva = X(va,:);
ytr_clean = Z_clean(tr);
yva_clean = Z_clean(va);

% Add noise to train and val set
sigma_train = 0.5 * P.sigma;
sigma_val   = 2.0 * P.sigma;

ytr = ytr_clean + sigma_train*randn(Ntr,1);
yva = yva_clean + sigma_val*randn(numel(va),1);

% Add additional noise and outliers to validation set
outlier_frac_val = 0.01;                               
kv = max(1, round(outlier_frac_val*numel(va)));
oi_val = randperm(numel(va), kv);
yva(oi_val) = yva(oi_val) + 3*std(yva) .* sign(randn(kv,1));

% Standardize base coordinates
muX = mean(Xtr,1); sX = std(Xtr,[],1)+1e-12;
muy = mean(ytr);   sy = std(ytr)+1e-12;

Xtrn = (Xtr - muX)./sX; 
Xvan = (Xva - muX)./sX;

ytrn = (ytr - muy)/sy;    
yvan = (yva - muy)/sy;

%===================================================================
% Radial Fourier features (for Qs3(k) only)
%[Xtrn, FF_meta] = fourier_features_radial(Xtrn, K_radial, P);
%Xvan            = fourier_features_radial_apply(Xvan, FF_meta);

%added segment for part(k)
%fprintf('total number of input features after transformation : %d\n',size(Xtrn,2));
%FrD = norm(Zgt - Zhat, 'fro');

%% ------------------ Sweep H (MBGD, constant LR) ------------------
best = struct('H',0,'W1',[],'b1',[],'W2',[],'b2',[], ...
              'rmse_tr',Inf,'rmse_va',Inf);
results = zeros(numel(H_list),3);

for i=1:numel(H_list)
    H = H_list(i);
    order_key = Xtrn(:,1);

    [W1,b1,W2,b2,logT] = train_mlp_mbgd( ...
        Xtrn,ytrn,H,max_epochs,eta,batch_size, ...
        Xvan,yvan,order_key);

    % eval on original scale
    yhat_tr = (tanh(Xtrn*W1 + b1)*W2 + b2)*sy + muy;
    yhat_va = (tanh(Xvan*W1 + b1)*W2 + b2)*sy + muy;
    rmse_tr = sqrt(mean((yhat_tr - ytr).^2));
    rmse_va = sqrt(mean((yhat_va - yva).^2));
    results(i,:) = [H rmse_tr rmse_va];

    if rmse_va < best.rmse_va
        best = struct('H',H,'W1',W1,'b1',b1,'W2',W2,'b2',b2, ...
                      'rmse_tr',rmse_tr,'rmse_va',rmse_va);
    end
    % Save log 
    write_csv_compat(sprintf('logs_H%03d_B%03d_mbgd.csv',H,batch_size), logT);
    end

fprintf('\n#Hidden\tRMSE_train\tRMSE_val\n');
disp(results);

%% ------------------ Plots: Ground truth vs Best Approx ------------------
Xall = [XX(:), YY(:)];
Xalln = (Xall - muX)./sX;

% Only for Qs3(k)
%Xalln = fourier_features_radial_apply(Xalln, FF_meta);

% Target Function
Zgt = f(Xall(:,1), Xall(:,2));

% Estimated Function (Inverse Normalization) 
Zph = tanh(Xalln*best.W1 + best.b1)*best.W2 + best.b2;
Zhat = Zph*sy + muy;

% Plot for target Function
% Fill in the blank
figure;
surf(XX,YY,reshape(Zgt,size(XX))); shading interp; grid on;
title('Target Function f_u(x,y)');
xlabel('x'); ylabel('y'); zlabel('f(x,y)');


%Plot for Estimated Function (MLP)
figure;
surf(XX,YY,reshape(Zhat,size(XX))); shading interp; grid on;
title(sprintf('MLP + Radial FF (MBGD, H=%d)', best.H));
xlabel('x'); ylabel('y'); zlabel('fhat');

%Plot for MSE curve
% Fill in the blank
figure;
plot(logT(:,2), 'LineWidth', 1.5); hold on;     % train MSE (column 2)
plot(logT(:,4), 'LineWidth', 1.5);              % val MSE (column 4)
xlabel('Epoch'); ylabel('MSE');
legend('Training MSE', 'Validation MSE');
title('MSE vs Epoch');
grid on;

%Plot for RMSE
% Fill in the blank
figure;
plot(logT(:,3),  'LineWidth', 1.5); hold on;     % Training RMSE (column 3)
plot(logT(:,5), 'LineWidth', 1.5);             % Validation RMSE (column 5)
xlabel('Epoch');
ylabel('RMSE');
legend('Training RMSE', 'Validation RMSE');
title('RMSE vs Epoch');
grid on;

% Gradient check
grad_check_small_batch(best.H, Xtrn, ytrn);

%% ======================================================================
% TRAINER (MBGD, constant LR)
function [W1,b1,W2,b2,logT] = train_mlp_mbgd( ...
        XtrnFF,ytrn,H,max_epochs,eta,B, XvanFF,yvanFF, order_key)

[N,D]=size(XtrnFF);
W1=0.5*randn(D,H); b1=zeros(1,H);
W2=0.5*randn(H,1); b2=0;
logT = zeros(max_epochs,7);

for e=1:max_epochs
    lr = eta;
    [~,p] = sort(order_key,'ascend');  % deterministic sweep
    gW1_last=0; gW2_last=0;

    for s = 1:B:N
        t = min(s+B-1, N);
        idx = p(s:t);
        x = XtrnFF(idx,:);  tvec = ytrn(idx,:); %target vector
        BT = size(x,1);

        % forward
        % Fill in the blank
        Z1 = x*W1+b1; %[BT x H]
        A1 = tanh(Z1); %[BT x H]
        yhat =A1*W2 +b2; %[BT x 1]
        diff = yhat-tvec; %[BT x 1]

        % backward
        % Fill in the blank
        dY  = (2/BT)*diff; %output error (derivative w.r.t yhat), [BT x 1]
        dW2 = A1'*dY; %[H x 1]
        db2 =sum(dY,1); %[1x1]
        dA1 = dY*W2'; %[BT x H]
        dZ1 = dA1.*(1-A1.^2); %[BT x H]
        dW1 = x'*dZ1; %[D x H]
        db1 = sum(dZ1,1); %[1 x H]

        % Parameters update
        W2 = W2 - lr*dW2; 
        b2 = b2-lr *db2;
        W1 = W1-lr *dW1; 
        b1 = b1 - lr *db1;

        gW1_last = norm(dW1,'fro'); gW2_last = norm(dW2,'fro');
    end

    % TRAIN metrics (standardized scale)
    yhat_tr = tanh(XtrnFF*W1 + b1)*W2 + b2;
    diff_tr = yhat_tr - ytrn;  
    mse_tr = mean(diff_tr.^2); 
    rmse_tr =sqrt(mean(diff_tr.^2));

    % VAL metrics (standardized scale)
    yhat_va = tanh(XvanFF*W1 + b1)*W2 + b2;
    diff_va = yhat_va - yvanFF;  
    mse_va = mean(diff_va.^2) ; 
    rmse_va =sqrt(mean(diff_va.^2));

    logT(e,:) = [e, mse_tr, rmse_tr, mse_va, rmse_va, gW1_last, gW2_last];
end
end
% Print results at the end of training
fprintf('Training complete: H=%d, B=%d, eta=%.1e\n', H, batch_size, eta);
fprintf('  Err_train (RMSE) = %.4f\n', rmse_tr);
fprintf('  Err_val   (RMSE) = %.4f\n\n', rmse_va);

%% ======================================================================
% GRADIENT CHECK on inputs
function grad_check_small_batch(H, XtrnFF, ytrn)
% Select 4 samples randonly
sel = randperm(size(XtrnFF,1), 4);
X = XtrnFF(sel,:); y = ytrn(sel);
D = size(X,2);

W1=0.1*randn(D,H); b1=zeros(1,H);
W2=0.1*randn(H,1); b2=0;

%===============================
%Fill in the blank, the code is the same as
%in train_mlp_mbgd()
% Forward
Z1=X*W1+b1; 
A1=tanh(Z1); 
yhat=A1*W2+b2;
B=size(X,1);

%Backward
dY=(2/B)*(yhat-y);
dW2=A1'*dY; 
db2=sum(dY,1);
dA1=dY*W2'; 
dZ1=dA1.*(1-A1.^2);
dW1=X'*dZ1;

h=1e-5;
   % ---- W1 ----
    k = 1;
    M = zeros(size(W1)); M(k) = 1;
    f1 = loss_fd_ff(X, y, W1+h*M, b1, W2, b2);
    f2 = loss_fd_ff(X, y, W1-h*M, b1, W2, b2);
    fd = (f1 - f2) / (2*h);
    an = dW1(k);
    rel = abs(fd - an) / max([1, abs(an), abs(fd)]);
    fprintf(' %-6s | %+14.6e | %+14.6e | %.3e\n', 'W1', an, fd, rel);

    % ---- W2 ----
    k = 1;
    M = zeros(size(W2)); M(k) = 1;
    f1 = loss_fd_ff(X, y, W1, b1, W2+h*M, b2);
    f2 = loss_fd_ff(X, y, W1, b1, W2-h*M, b2);
    fd = (f1 - f2) / (2*h);
    an = dW2(k);
    rel = abs(fd - an) / max([1, abs(an), abs(fd)]);
    fprintf(' %-6s | %+14.6e | %+14.6e | %.3e\n', 'W2', an, fd, rel);

    % ---- b1 ----
    k = 1;
    M = zeros(size(b1)); M(k) = 1;
    f1 = loss_fd_ff(X, y, W1, b1+h*M, W2, b2);
    f2 = loss_fd_ff(X, y, W1, b1-h*M, W2, b2);
    fd = (f1 - f2) / (2*h);
    an = b1(k);
    rel = abs(fd - an) / max([1, abs(an), abs(fd)]);
    fprintf(' %-6s | %+14.6e | %+14.6e | %.3e\n', 'b1', an, fd, rel);

    % ---- b2 ----
    f1 = loss_fd_ff(X, y, W1, b1, W2, b2+h);
    f2 = loss_fd_ff(X, y, W1, b1, W2, b2-h);
    fd = (f1 - f2) / (2*h);
    an = db2;
    rel = abs(fd - an) / max([1, abs(an), abs(fd)]);
    fprintf(' %-6s | %+14.6e | %+14.6e | %.3e\n', 'b2', an, fd, rel);
    
end

function L=loss_fd_ff(X,y,W1,b1,W2,b2)
Z1=X*W1+b1; 
A1=tanh(Z1); 
yhat=A1*W2+b2;
L=mean((yhat - y).^2);
end

%% ======================================================================
% Radial Fourier features
function [Phi, meta] = fourier_features_radial(X2d, K, P)
% X2d standardized; K = #radial harmonics
% P = personalized parameters
N = size(X2d,1);
% Fill in the blank
% convert (x, y) to polar system (r, theta) 
x= X2d(:,1);
y=X2d(:,2);
r = sqrt(x.^2+y.^2);
theta = atan2(y,x);
omega_star = 12 * P.alpha_u;             % dominant radial frequency
omegas = linspace(0.25*omega_star, 2.5*omega_star, K);

Phi = ones(N,1);                         % bias
Phi = [Phi, r];                          % low-freq radius

env1 = 1 ./ (P.alpha_u*(r.^2) + (2 + P.c_u));
env2 = env1.^2;
Phi = [Phi, env1, env2];

Blk = zeros(N, 2*K);
col=1;

%=======================================
for w = omegas
    Blk(:,col) = sin(w*r); col=col+1;
    Blk(:,col) = cos(w*r); col=col+1;
end

Phi = [Phi, Blk];

Phi = [Phi, cos(theta), sin(theta)];
%======================================

meta.kind   = 'radial';
meta.omegas = omegas;
meta.P      = P;
end

function Phi = fourier_features_radial_apply(X2d, meta)
% convert (x, y) to polar system (r, theta) 
% Fill in the blank
x= X2d(:,1);
y=X2d(:,2);
r = sqrt(x.^2+y.^2);
theta = atan2(y,x);
P = meta.P; omegas = meta.omegas;

Phi = ones(size(r));
Phi = [Phi, r];

env1 = 1 ./ (P.alpha_u*(r.^2) + (2 + P.c_u));
env2 = env1.^2;
Phi = [Phi, env1, env2];

Blk = zeros(numel(r), 2*numel(omegas));
col=1;
for w = omegas
    Blk(:,col) = sin(w*r); col=col+1;
    Blk(:,col) = cos(w*r); col=col+1;
end
Phi = [Phi, Blk];
Phi = [Phi, cos(theta), sin(theta)];end

%% ======================================================================
% UTIL: write CSV
function write_csv_compat(fname, M)
if exist('writematrix','file') == 2
    writematrix(M, fname);
else
    csvwrite(fname, M);
end
end

%% ======================================================================
% STUDENT-ID -> parameters (seed, alpha_u, c_u, sigma)
function P = params_from_student_id(student_id)
d = double(student_id) - 48; d = d(d>=0 & d<=9);
if isempty(d), error('ID must contain digits.'); end
s0 = uint64(0);
for i=1:numel(d)
    s0 = s0 + uint64(d(i)) * uint64(131)^(i-1);
end
s = mod(s0, 2^32);
[s,u1] = lcg_step(s);
[s,u2] = lcg_step(s);
[~,u3] = lcg_step(s);
P.alpha_u = 0.8 + 0.8*u1;
P.c_u     = -0.3 + 0.6*u2;
P.sigma   = 0.02 + 0.06*u3;
P.seed    = uint32(mod(double(s0), 2^31 - 1) + 1);
end

function [s,u] = lcg_step(s)
s = uint64(mod(1664525*double(s) + 1013904223, 2^32));
u = double(s)/double(2^32);
end



%added segment for part(k) 
% --- Print results clearly ---
%fprintf('\nFrobenius distance (FrD) = %.4f\n', FrD);
%fprintf('Best model: H* = %d , RMSE_train = %.4f , RMSE_val = %.4f\n ' ,...
%        best.H, best.rmse_tr, best.rmse_va);

%code for Qs3(h) --> Col 6 and 7 store norm dW1 and norm dW2
figure;
plot(logT(:,6), 'LineWidth', 1.5); hold on;
plot(logT(:,7), 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Gradient norm (Frobenius)');
legend('||∇W1||','||∇W2||');
title('Gradient Norms vs Epoch');
grid on;
