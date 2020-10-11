function HMModel = HMMLearningCompleteData(TrainingStructure, HMModel)
nSeq = size(TrainingStructure, 2);
%z = labels;
%N = numel(y);
K = HMModel.StateParameters.K ;
iIni = HMModel.ARorder + 1;
%% Format input
for i = 1:nSeq
    y = TrainingStructure(i).y;
    % Observations must be row vectors
    if iscolumn(y)
        y = y';
    end
    if HMModel.normalize
        y = zscore(y, [], 2);                          % Normalize observations
    end
    TrainingStructure(i).y = y;
    TrainingStructure(i).N = size(y, 2);
end
%% Parameters of observtion model, i.e. emissions
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED')
    % Univariate Gaussian case - TODO multiple training sequences
    if strcmpi(HMModel.ObsParameters.model, 'Gaussian')
        yall = horzcat(TrainingStructure.y);
        labelsall = horzcat(TrainingStructure.z);
        for k = 1:K
            HMModel.ObsParameters.mu(k) = mean(yall(labelsall == k));
            HMModel.ObsParameters.sigma(k) = std(yall(labelsall == k), 1);
        end
    elseif strcmpi(HMModel.ObsParameters.model, 'MultivariateGaussian')
        yall = horzcat(TrainingStructure.y);
        labelsall = horzcat(TrainingStructure.z);
        for k = 1:K
            aux = yall(:, labelsall == k);
            HMModel.ObsParameters.mu(:, k) = mean(aux, 2);
            HMModel.ObsParameters.sigma(:, :, k) = cov(aux');
        end
    end
elseif strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED')
    p = HMModel.ARorder;
    Ypcell = cell(1, nSeq);
    labelsall = [];
    yauxall = [];
    for train_i = 1:nSeq
        y = TrainingStructure(train_i).y;
        N = TrainingStructure(train_i).N;
        % Build auxiliary matrix of AR predictors
        Yp = zeros(N - p, p);
        for i = 1:N-p
            Yp(i, :) = -fliplr(y(i : i + p - 1));
        end  
        Ypcell{train_i} = Yp;
        labelsall = [labelsall TrainingStructure(train_i).z(iIni:end)];
        yauxall = [yauxall TrainingStructure(train_i).y(iIni:end)];
    end
    clear Yp
    Yp = cell2mat(Ypcell');
    ak = zeros(p, K);
    sigk = zeros(1, K);
    for k = 1:K
        idx = find(labelsall == k);
        Ytil = yauxall(idx)';
        Xtil = Yp(idx, :)';
        if HMModel.robustMstep == 1
            % Robust fit
            %[ak(:, k), stats] = robustfit(Xtil', Ytil, 'bisquare', [], 'off');
            %sigk(k) = stats.robust_s;
            mdl = fitlm(Xtil', Ytil', 'Intercept', false, 'RobustOpts', 'welsch'); 
            HMModel.ObsParameters.meanFcn(k).model = mdl;
            err = Xtil'*table2array(mdl.Coefficients(:,1)) - Ytil;
        else
            if ~isfield(HMModel.ObsParameters, 'meanModel')
                HMModel.ObsParameters.meanModel = 'Linear';
            end
            
            % Normalize
            %Xtil = normalize(Xtil, 2, 'zscore');
            %Ytil = normalize(Ytil, 1, 'zscore');
            
            if strcmpi(HMModel.ObsParameters.meanModel, 'Linear')
                % OLS
                %ak(:, k) = (Xtil*Xtil')\(Xtil*Ytil);
                %err = Xtil'*ak(:, k) - Ytil;
                %sigk(k) = std(err, 1);
                mdl = fitlm(Xtil', Ytil', 'Intercept', false); 
                HMModel.ObsParameters.meanFcn(k).model = mdl;
                err = Xtil'*table2array(mdl.Coefficients(:,1)) - Ytil;
            elseif strcmpi(HMModel.ObsParameters.meanModel, 'MLP')
                net = feedforwardnet(ceil(p/2), 'trainbr');
                %net = fitnet(ceil(p/2), 'trainlm');
                net = configure(net, Xtil, Ytil');
                net.trainParam.showWindow = 0;
                net = train(net, Xtil, Ytil');
                HMModel.ObsParameters.meanFcn(k).model = net;
                err = net(Xtil)' - Ytil;
                %mdl = fitrensemble(Xtil, Ytil, 'Method', 'Bag');
            elseif strcmpi(HMModel.ObsParameters.meanModel, 'SVM')
                mdl = fitrkernel(Xtil', Ytil, 'NumExpansionDimensions', p^2, 'Learner', 'leastsquares');
                %mdl = fitrsvm(Xtil', Ytil, 'KernelFunction', 'gaussian', 'RemoveDuplicates', true, 'ShrinkagePeriod', 1000);
                HMModel.ObsParameters.meanFcn(k).model = mdl;
                err = predict(mdl, Xtil') - Ytil;
            end
        end
        % Residual model, i.e. errors
        if strcmpi(HMModel.ObsParameters.model, 'Gaussian')
            HMModel.ObsParameters.sigma(k) = std(err, 1);
        elseif strcmpi(HMModel.ObsParameters.model, 'Generalizedt')
            pd = fitdist(err, 'tLocationScale');
            HMModel.ObsParameters.sigma(k) = pd.sigma;
            HMModel.ObsParameters.nu(k) = pd.nu;
        end
    end
    %HMModel.ObsParameters.ARcoeff = ak;
    %HMModel.ObsParameters.sigma = sigk;
end
%% Parameters of hidden markov chain
HMModel.StateParameters.pi = zeros(K, 1);
% only works for K = 2 and sleep spindles so far, TODO
HMModel.StateParameters.pi(1) = 1;

if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
    A = zeros(K, K);
    for train_i = 1:nSeq
        N = TrainingStructure(train_i).N;
        labels = TrainingStructure(train_i).z;
        for i = iIni:N - 1                         % only works for K = 2 so far, TODO
            if labels(i) == 1 && labels(i + 1) == 1
                A(1, 1) = A(1, 1) + 1;
            elseif labels(i) == 1 && labels(i + 1) == 2
                A(1, 2) = A(1, 2) + 1;
            elseif labels(i) == 2 && labels(i + 1) == 1
                A(2, 1) = A(2, 1) + 1;
            elseif labels(i) == 2 && labels(i + 1) == 2
                A(2, 2) = A(2, 2) + 1;
            end
        end
    end
    for i = 1:2
        A(i, :) = A(i, :)./sum(A(i, :));
    end
    HMModel.StateParameters.A = A;
elseif strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
    % Estimate both hidden markov chain and duration parameters
    if ~isfield(HMModel.DurationParameters, 'dmax')
        % no dmax provided - estimate from labels
        dmink = zeros(nSeq, K);
        dmaxk = zeros(nSeq, K);
        ModeStart = cell(nSeq, K);
        ModeEnd = cell(nSeq, K);
        for train_i = 1:nSeq
            N = TrainingStructure(train_i).N;
            labels = TrainingStructure(train_i).z;           
            for k = 1:K
                idx = labels == k;
                if labels(iIni) == k
                    aux1 = [iIni find(diff(idx) == -1)];
                    aux2 = find(diff(idx) == 1);
                else
                    aux1 = find(diff(idx) == 1);
                    aux2 = find(diff(idx) == -1);
                end
                if numel(aux1) > numel(aux2)
                    aux2 = [aux2 N];
                end
                ModeStart{train_i, k} = aux1;
                ModeEnd{train_i, k} = aux2;
                daux = aux2 - aux1;
                dmink(train_i, k) = min(daux);
                dmaxk(train_i, k) = max(daux);
            end
            clear aux1 aux2
        end
        dmin = min(dmink(:));
        dmax = max(dmaxk(:));
        for k = 1:K
            daux = cell2mat(ModeEnd(:,k)') - cell2mat(ModeStart(:,k)');
            %daux = ModeEnd{k} - ModeStart{k};
            [f, ~] = ksdensity(daux, 1:dmax);
            HMModel.DurationParameters.PNonParametric(k, :) = f;
        end
        HMModel.DurationParameters.dmax = dmax;
        HMModel.DurationParameters.dmin = dmin;
        
        % Transition matrices
        HMModel.StateParameters.A(:,:,1) = [0 1;1 0];           % only works for K = 2       TODO
        HMModel.StateParameters.A(:,:,2) = eye(K);
    else
        % dmax provided - only works for K = 2                  TODO
        dmax = HMModel.DurationParameters.dmax;
        dState = cell(nSeq, K);
        A = zeros(K, K);
        for train_i = 1:nSeq
            fl = 1;
            i = iIni;
            N = TrainingStructure(train_i).N;
            z = TrainingStructure(train_i).z;
            while fl
                aux = z(i: min([i+dmax-1, N]));
                idx1 = find(aux == 2, 1);
                if ~isempty(idx1)
                    if idx1 == 1
                        A(1, 2) = A(1, 2) + 1;
                        idx2 = find(z(i+idx1-1:end) == 1, 1);
                        A(2, 1) = A(2, 1) + 1;
                        dState{train_i, 2} = [dState{train_i, 2} idx2 - 1];
                        i = idx1 + idx2 + i - 2;                       
                    else
                        A(1, 2) = A(1, 2) + 1;
                        dState{train_i, 1} = [dState{train_i, 1} idx1-1];
                        idx2 = find(z(i+idx1-1:end) == 1, 1);
                        A(2, 1) = A(2, 1) + 1;
                        dState{train_i, 2} = [dState{train_i, 2} idx2 - 1];
                        i = idx1 + idx2 + i - 2;     
                    end      
                else
                    dState{train_i, 1} = [dState{train_i, 1} dmax];
                    A(1, 1) = A(1, 1) + 1;
                    i = i + dmax;
                end                           
                if i > numel(z)
                    break
                end
            end
        end
        
        dmink = zeros(1, K);
        dmaxk = zeros(1, K);
        for k = 1:K
            aux = cell2mat(dState(:,k)');
            aux = aux(aux >= HMModel.DurationParameters.dmin);
            dmink(k) = min(aux);
            dmaxk(k) = max(cell2mat(dState(:,k)'));
        end
        dmin = min(dmink);
        dmax = max(dmaxk);
        for k = 1:K
            daux = cell2mat(dState(:,k)');
            daux = daux(daux >= HMModel.DurationParameters.dmin);
            [f, ~] = ksdensity(daux, dmin:dmax);
            HMModel.DurationParameters.PNonParametric(k, :) = [zeros(1, dmin-1) f];
        end
        HMModel.DurationParameters.dmax = dmax;
        HMModel.DurationParameters.dmin = dmin;
        
        for i = 1:2
            A(i, :) = A(i, :)./sum(A(i, :));
        end
        HMModel.StateParameters.A(:, :, 1) = A;
        HMModel.StateParameters.A(:,:,2) = eye(K);
    end
end

% %% Parameters of duration model
% if nargin == 3
%     % no dmax provided
%     dmink = zeros(1, K);
%     dmaxk = zeros(1, K);
%     ModeStart = cell(1, K);
%     ModeEnd = cell(1, K);
%     for k = 1:K
%         idx = labels == k;
%         if labels(iIni) == k
%             aux1 = [iIni find(diff(idx) == -1)];
%             aux2 = find(diff(idx) == 1);
%         else
%             aux1 = find(diff(idx) == 1);
%             aux2 = find(diff(idx) == -1);
%         end
%         if numel(aux1) > numel(aux2)
%             aux2 = [aux2 numel(y)];
%         end
%         ModeStart{k} = aux1;
%         ModeEnd{k} = aux2;
%         daux = aux2 - aux1;
%         dmink(k) = min(daux);
%         dmaxk(k) = max(daux);
%     end
%     dmin = min(dmink);
%     dmax = max(dmaxk);
%     edges = 0.5:1:dmax+0.5;
%     for k = 1:K
%         daux = ModeEnd{k} - ModeStart{k};
%         %[N, ~] = histcounts(daux, edges);
%         
%         [f, ~] = ksdensity(daux, 1:dmax);
%         %dcount(k, :) = f;
%         
%         %HMModel.DurationParameters.PNonParametric(k, :) = N/sum(N);
%         HMModel.DurationParameters.PNonParametric(k, :) = f;
%     end
%     HMModel.DurationParameters.dmax = dmax;
%     HMModel.DurationParameters.dmin = dmin;
% else
%     % dmax provided - only works for K = 2                  TODO
%     dState = cell(1, K);fl = 1;
%     i = iIni;
%     while fl
%         aux = z(i: min([i+dmax-1, numel(y)]));
%         idx1 = find(aux == 2, 1);
%         if ~isempty(idx1)
%             dState{1} = [dState{1} idx1-1];
%             %idx2 = find(aux(idx1:end) == 1, 1);
%             idx2 = find(z(i+idx1-1:end) == 1, 1);
%             dState{2} = [dState{2} idx2 - 1];
%             i = idx1 + idx2 + i - 2;
%         else
%             dState{1} = [dState{1} dmax];
%             i = i + dmax;
%         end
%         if i > numel(z)
%             break
%         end
%     end
%     dmink = zeros(1, K);
%     dmaxk = zeros(1, K);
%     for k = 1:K
%         dmink(k) = min(dState{k});
%         dmaxk(k) = max(dState{k});
%     end
%     dmin = min(dmink);
%     dmax = max(dmaxk);
%     for k = 1:K
%         daux = dState{k};
%         [f, ~] = ksdensity(daux, 1:dmax);
%         HMModel.DurationParameters.PNonParametric(k, :) = f;
%     end
%     HMModel.DurationParameters.dmax = dmax;
%     HMModel.DurationParameters.dmin = dmin;
% end
% 
% if strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
%     %HMModel.StateParameters.A(:,:,1) = [0 1;1 0];           % only works for K = 2       TODO
%     HMModel.StateParameters.A(:,:,2) = eye(K);
% end
% 
% if strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED')
%     HMModel = rmfield(HMModel, 'DelayARMatrix');
% end

end