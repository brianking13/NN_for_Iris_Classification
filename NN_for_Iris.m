clear;clc;

% Data(instance, parameter)- p1,p2,p3,p4,label
data = split(string(importdata('iris-original.data',',')),",");

% Split data
ID1 = data(1:50,1:5);
ID2 = data(51:100,1:5);
ID3 = data(101:150,1:5);
training = cat(1,ID1(1:40,:),ID2(1:40,:),ID3(1:40,:));
test = cat(1,ID1(41:50,:),ID2(41:50,:),ID3(41:50,:));

[features,t] = prepare_data(training);
[features_training, tt] = prepare_data(test);

% Plot Features
figure(1)
subplot(2,2,1)
plot_sets(features,1,2)

subplot(2,2,2)
plot_sets(features,1,3)

subplot(2,2,3)
plot_sets(features,1,4)

subplot(2,2,4)
plot_sets(features,3,4)

% Prepare data for Neural Network

% Training Data
x(1,:) = rescale(features(:,1))';
x(2,:) = rescale(features(:,2))';
x(3,:) = rescale(features(:,3))';
x(4,:) = rescale(features(:,4))';

% Testing Data
xt(1,:) = rescale(features_training(:,1))';
xt(2,:) = rescale(features_training(:,2))';
xt(3,:) = rescale(features_training(:,3))';
xt(4,:) = rescale(features_training(:,4))';

output = t;

% Neural Network Architecture Variables
ins = 4; % Number of input nodes
hids = 6; % Number of hidden nodes
outs = 3; % Number of output nodes
examples = length(x(1,:)); % Number of examples

% Learning Parameters
kappa = 0.1;
phi = 0.5;
theta = 0.7;
mu = .9;

% Weights and Derivatives
a(1:(ins+1), 1:hids) = 0.0; % Hidden weights
b(1:(hids+1), 1:outs) = 0.0; % Output weights
cHid(1:(ins+1), 1:hids) = 0.0; % Weight changes
cOut(1:(hids+ins+1), 1:outs) = 0.0;
dHid(1:(ins+1), 1:hids) = 0.0; % Derivatives
dOut(1:(hids+ins+1), 1:outs) = 0.0;
eHid(1:(ins+1), 1:hids) = 0.0; % Adaptive learning rates
eOut(1:(hids+ins+1), 1:outs) = 0.0;
fHid(1:(ins+1), 1:hids) = 0.0; % Recent average of derivatives
fOut(1:(hids+ins+1), 1:outs) = 0.0;
u = 0.0; % Weighted sum for hidden node 
y(1:hids) = 0.0; % Hidden node outputs
v = 0.0; % Weighted sum for output node
z(1:outs) = 0.0; % Output node outputs
p(1:outs) = 0.0; % dE/dv
epoch = 0; % Current epoch number


%initWeights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1:hids,
    for i = 1:(ins+1),
        a(i,j) = 0.2 * (rand - 0.5);
        eHid(i,j) = kappa;
    end % i
end % j
for k = 1:outs,
    for j = 1:(hids+1),
        if mod(j,2) == 0
            b(j,k) = 1;
        else
            b(j,k) = -1;
        end % if
    eOut(j,k) = kappa;
   end % j
end % k

num_epochs = 5000;
loading = 0;
% The main program
while (epoch < num_epochs)
    loading = loading+1;
    if mod(epoch,250) ==0
        clc
        disp("Loading: " +loading/num_epochs*100+"%")
    elseif loading == num_epochs
        clc
        disp("Loading: 100%")
    end
    epoch = epoch + 1;
    error_sum(epoch) = 0;
    for n = 1:examples % Cycling through examples for epoch-based update

         %Forward Evaluation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for j = 1:hids,
            u = a(ins+1,j); % Bias weight
            for i = 1:ins,
                u = u + (a(i,j) * x(i,n)); % Weighted inputs
            end % i
            y(j) = logistic(u);
           
        end % j
        for k = 1:outs,
            v = b(hids+1,k); % Bias weight
            for j = 1:hids,
                v = v + (b(j,k) * y(j)); % Hidden outputs, wtd.
            end % j
            z(k) = logistic(v);
            
        end % k
     
        [A,I] = max(z);
        [B,t_index] = max(t(n,:));
        s(1,n) = I; % For recording results
 
        if t_index ~= I 
            error_sum(epoch) = error_sum(epoch) + 1;% Sum of errors
        end
        


        % Backpropagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        q(1:hids) = 0.0; % dE/du (reset to 0 each time)
        for k = 1:outs,% Calculate error derivatives on output node weights
            p(k) = (z(k) - output(n,k)) * z(k) * (1 - z(k));
            dOut(hids+1,k) = dOut(hids+1,k) + p(k); % Bias weight
            for j = 1:hids, % Weights on hids
                dOut(j,k) = dOut(j,k) + p(k) * y(j);
                q(j) = q(j) + p(k) * b(j,k); % Used below
            end % j
        end % k
        for j = 1:hids,
            q(j) = q(j) * y(j) * (1 - y(j));
            dHid(ins+1,j) = dHid(ins+1,j) + q(j); % Bias weight
            for i = 1:ins, % Weights on ins
                dHid(i,j) = dHid(i,j) + q(j) * x(i,n);
            end % i
        end % j


    end % n - examples
    

   % Change Weights %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:hids, % Change weights on hidden nodes
        for i = 1:(ins+1),
            if dHid(i,j)*fHid(i,j) > 0.0 
                eHid(i,j) = eHid(i,j) + kappa;
            else
                eHid(i,j) = eHid(i,j) * phi;
            end
            fHid(i,j) = theta * fHid(i,j) + (1 - theta) * dHid(i,j);
            cHid(i,j) = mu *cHid(i,j) - (1 - mu) * eHid(i,j) * dHid(i,j);
            a(i,j) = a(i,j) + cHid(i,j);
        end % i
    end % j
    for k = 1:outs, % Change weights on output nodes
        for j = 1:(hids+1),
            if dOut(j,k)*fOut(j,k) > 0.0 
                eOut(j,k) = eOut(j,k) + kappa;
            else
                eOut(j,k) = eOut(j,k) * phi;
            end
            fOut(j,k) = theta * fOut(j,k) + (1 - theta) * dOut(j,k);
            cOut(j,k) = mu *cOut(j,k) - (1 - mu) * eOut(j,k) * dOut(j,k);
            b(j,k) = b(j,k) + cOut(j,k);
        end % j
    end % k
    dHid(1:(ins+1), 1:hids) = 0; % Reset to 0
    dOut(1:(hids+1), 1:outs) = 0; % Reset to 0

    dHid(1:(ins+1), 1:hids) = 0; % Reset to 0
    dOut(1:(hids+1), 1:outs) = 0; % Reset to 0
end % while

figure(2)
plot(error_sum,'*');
xlabel('Epoch'); ylabel('Error');


%% test newtork
output = tt;
 for n = 1:30 % Cycling through examples for epoch-based update

         %Forward Evaluation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for j = 1:hids,
            u = a(ins+1,j); % Bias weight
            for i = 1:ins,
                u = u + (a(i,j) * xt(i,n)); % Weighted inputs
            end % i
            y(j) = logistic(u);
           
        end % j
        for k = 1:outs,
            v = b(hids+1,k); % Bias weight
            for j = 1:hids,
                v = v + (b(j,k) * y(j)); % Hidden outputs, wtd.
            end % j
            z(k) = logistic(v);         
            
        end % k
        [A,I] = max(z);
        [B,t_index] = max(tt(n,:));
        st(n,:) = z; % For recording results
        if t_index ~= I 
            error_sum(epoch) = error_sum(epoch) + 1;% Sum of errors
        end
      

 end % n - examples

 figure(3)
 plot(1:30, st, 'o')
 legend('ID1','ID2','ID3')
 
%% functions
function [features, t]=prepare_data(data)
    % create feature matrix
    features = double(data(:,1:4));

    % create label matrix
    for i = 1:length(data)
        if data(i,5) == "Iris-setosa"
            t(i,1:3) = [1,0,0];
        elseif data(i,5) == "Iris-versicolor"
            t(i,1:3) = [0,1,0];
        else
            t(i,1:3) = [0,0,1];
        end
    end
end

function [] = plot_sets(feature_matrix, f1,f2)
    scatter(feature_matrix(1:40,f1), feature_matrix(1:40,f2),'r')
    hold on
    scatter(feature_matrix(41:80,f1), feature_matrix(41:80,f2),'g')
    hold on
    scatter(feature_matrix(81:120,f1), feature_matrix(81:120,f2),'b')
    hold on
    % legend(label_matrix(1),label_matrix(41),label_matrix(81))
    xlabel('Feature ' + string(f1))
    ylabel('Feature ' + string(f2))
end
