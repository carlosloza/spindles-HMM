function HMModel = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel)
nSeq = size(TrainingStructure, 2);
K = 2;
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
            mdl = fitlm(Xtil', Ytil', 'Intercept', false, 'RobustOpts', 'welsch'); 
            HMModel.ObsParameters.meanParameters(k).Coefficients = table2array(mdl.Coefficients(:,1));
            err = Xtil'*HMModel.ObsParameters.meanParameters(k).Coefficients - Ytil;
        else
%             if ~isfield(HMModel.ObsParameters, 'meanModel')
%                 HMModel.ObsParameters.meanModel = 'Linear';
%             end
            if strcmpi(HMModel.ObsParameters.meanModel, 'Linear')
                if HMModel.robustMstep
                    mdl = fitlm(Xtil', Ytil', 'Intercept', false, 'RobustOpts', 'welsch'); 
                else
                    mdl = fitlm(Xtil', Ytil', 'Intercept', false); 
                end
                
                HMModel.ObsParameters.meanParameters(k).Coefficients = table2array(mdl.Coefficients(:,1));               
                err = Xtil'*HMModel.ObsParameters.meanParameters(k).Coefficients - Ytil;
            elseif strcmpi(HMModel.ObsParameters.meanModel, 'MLP')
                net = feedforwardnet(ceil(p/2), 'trainbr');
                %net = fitnet(ceil(p/2), 'trainlm');
                net = configure(net, Xtil, Ytil');
                net.trainParam.showWindow = 0;
                net = train(net, Xtil, Ytil');
                HMModel.ObsParameters.meanFcn(k).model = net;
                err = net(Xtil)' - Ytil;
            elseif strcmpi(HMModel.ObsParameters.meanModel, 'SVM')
                mdl = fitrkernel(Xtil', Ytil, 'NumExpansionDimensions', p^2, 'Learner', 'leastsquares');
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
end

% [HH, FF] = freqz(1, [1; HMModel.ObsParameters.meanParameters(1).Coefficients], 1024, HMModel.Fs);
% plot(FF, 20*log10(abs(HH)))
% hold on
% [HH, FF] = freqz(1, [1; HMModel.ObsParameters.meanParameters(2).Coefficients], 1024, HMModel.Fs);
% plot(FF, 20*log10(abs(HH)), 'r')
% pause(0.1)

%% Parameters of hidden markov chain
HMModel.StateParameters.pi = [1 0]';

if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
    A = zeros(K, K);
    for train_i = 1:nSeq
        N = TrainingStructure(train_i).N;
        labels = TrainingStructure(train_i).z;
        for i = iIni:N - 1                         
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
        % No dmax provided
        A = [0 1;1 0];
        z1 = zeros(1, N);
        z2 = zeros(1, N);
        for train_i = 1:nSeq
            i = iIni;
            N = TrainingStructure(train_i).N;
            z = TrainingStructure(train_i).z;
            % BEWARE: this is clunky
            ct = 1;
            for i = 2:N
                % durations
                if z(i) == z(i-1)
                    ct = ct + 1;
                else
                    if z(i-1) == 1
                        z1(ct) = z1(ct) + 1;
                    elseif z(i-1) == 2
                        z2(ct) = z2(ct) + 1;
                    end
                    ct = 1;
                end
            end
        end
        idx = find(z1 ~= 0);
        HMModel.DurationParameters.dmax = idx(end);
        z1 = z1(1: HMModel.DurationParameters.dmax);
        z2 = z2(1: HMModel.DurationParameters.dmax);
        z2Seq = [];
        idxz2 = find(z2 > 0);
        for i = 1:numel(idxz2)
            z2Seq = [z2Seq idxz2(i)*ones(1, z2(idxz2(i)))];
        end
        HMModel.DurationParameters.PNonParametric(1, :) = z1./sum(z1);
        HMModel.DurationParameters.PNonParametric(2, :) = z2./sum(z2);
        % This is useful for initial conditions of EM
        HMModel.DurationParameters.Ini = [median(z2Seq) mad(z2Seq, 0)];      
    else
        % dmax is provided
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
        dmin = HMModel.DurationParameters.dmin;
        for k = 1:K
            zSeq = cell2mat(dState(:,k)');
            zSeq(zSeq < dmin) = dmin;
            counts = histcounts(zSeq, 1:HMModel.DurationParameters.dmax + 1);
            HMModel.DurationParameters.PNonParametric(k, :) = counts./sum(counts);
            if k == 2
                HMModel.DurationParameters.Ini = [median(zSeq) mad(zSeq, 0)];
            end
        end
        for i = 1:2
            A(i, :) = A(i, :)./sum(A(i, :));
        end
        
    end   
%     if ~isfield(HMModel.DurationParameters, 'dmax')
%         % no dmax provided - estimate from labels
%         dmink = zeros(nSeq, K);
%         dmaxk = zeros(nSeq, K);
%         ModeStart = cell(nSeq, K);
%         ModeEnd = cell(nSeq, K);
%         for train_i = 1:nSeq
%             N = TrainingStructure(train_i).N;
%             labels = TrainingStructure(train_i).z;           
%             for k = 1:K
%                 idx = labels == k;
%                 if labels(iIni) == k
%                     aux1 = [iIni find(diff(idx) == -1)];
%                     aux2 = find(diff(idx) == 1);
%                 else
%                     aux1 = find(diff(idx) == 1);
%                     aux2 = find(diff(idx) == -1);
%                 end
%                 if numel(aux1) > numel(aux2)
%                     aux2 = [aux2 N];
%                 end
%                 ModeStart{train_i, k} = aux1;
%                 ModeEnd{train_i, k} = aux2;
%                 daux = aux2 - aux1;
%                 dmink(train_i, k) = min(daux);
%                 dmaxk(train_i, k) = max(daux);
%             end
%             clear aux1 aux2
%         end
%         dmin = min(dmink(:));
%         dmax = max(dmaxk(:));
%         for k = 1:K
%             daux = cell2mat(ModeEnd(:,k)') - cell2mat(ModeStart(:,k)');
%             %daux = ModeEnd{k} - ModeStart{k};
%             
%             [f, ~] = ksdensity(daux, 1:dmax);
%             f = f/sum(f);
%             
%             HMModel.DurationParameters.PNonParametric(k, :) = f;
%         end
%         HMModel.DurationParameters.dmax = dmax;
%         HMModel.DurationParameters.dmin = dmin;
%         
%         % Transition matrices
%         HMModel.StateParameters.A(:,:,1) = [0 1;1 0];           
%         HMModel.StateParameters.A(:,:,2) = eye(K);
%    else
% dmax provided
%dmax = HMModel.DurationParameters.dmax;
%dState = cell(nSeq, K);


% A = [0 1;1 0];
% z1 = zeros(1, N);
% z2 = zeros(1, N);
% for train_i = 1:nSeq
%     fl = 1;
%     i = iIni;
%     N = TrainingStructure(train_i).N;
%     z = TrainingStructure(train_i).z;
%     % BEWARE: this is clunky
%     ct = 1;
%     for i = 2:N
%         % durations
%         if z(i) == z(i-1)
%             ct = ct + 1;
%         else
%             if z(i-1) == 1
%                 z1(ct) = z1(ct) + 1;
%             elseif z(i-1) == 2
%                 z2(ct) = z2(ct) + 1;
%             end
%             ct = 1;
%         end
%     end
    
%     while fl
%         aux = z(i: min([i+dmax-1, N]));
%         idx1 = find(aux == 2, 1);
%         if ~isempty(idx1)
%             if idx1 == 1
%                 A(1, 2) = A(1, 2) + 1;
%                 idx2 = find(z(i+idx1-1:end) == 1, 1);
%                 A(2, 1) = A(2, 1) + 1;
%                 dState{train_i, 2} = [dState{train_i, 2} idx2 - 1];
%                 i = idx1 + idx2 + i - 2;
%             else
%                 A(1, 2) = A(1, 2) + 1;
%                 dState{train_i, 1} = [dState{train_i, 1} idx1-1];
%                 idx2 = find(z(i+idx1-1:end) == 1, 1);
%                 A(2, 1) = A(2, 1) + 1;
%                 dState{train_i, 2} = [dState{train_i, 2} idx2 - 1];
%                 i = idx1 + idx2 + i - 2;
%             end
%         else
%             dState{train_i, 1} = [dState{train_i, 1} dmax];
%             A(1, 1) = A(1, 1) + 1;
%             i = i + dmax;
%         end
%         if i > numel(z)
%             break
%         end
%     end


% if ~isfield(HMModel.DurationParameters, 'dmax')
%     idx = find(z1 ~= 0);
%     HMModel.DurationParameters.dmax = idx(end);
% end
% z1 = z1(1: HMModel.DurationParameters.dmax);
% z2 = z2(1: HMModel.DurationParameters.dmax);
% z2Seq = [];
% idxz2 = find(z2 > 0);
% for i = 1:numel(idxz2)
%     z2Seq = [z2Seq idxz2(i)*ones(1, z2(idxz2(i)))];
% end
% HMModel.DurationParameters.PNonParametric(1, :) = z1./sum(z1);
% HMModel.DurationParameters.PNonParametric(2, :) = z2./sum(z2);
% HMModel.DurationParameters.Ini = [median(z2Seq) mad(z2Seq, 0)];

% %% All this applies to Fs = 100 Hz only so be careful
% dmink = zeros(1, K);
% dmaxk = zeros(1, K);
% dStateall = cell(1, K);
% histint = 1;
% for k = 1:K
%     aux = cell2mat(dState(:,k)');
%     if k == 1
%         N = histcounts(aux, HMModel.ARorder+1:histint:HMModel.DurationParameters.dmax);
%         perc = round(0.1*sum(N))/sum(N == 0);
%         N(N == 0) = perc;    
%         N = N/sum(N);
%         HMModel.DurationParameters.PNonParametric(k, :) = [zeros(1, HMModel.ARorder+1) N];
%     else
%         N = histcounts(aux, HMModel.DurationParameters.dmin:histint:HMModel.DurationParameters.dmax);
%         perc = round(0.1*sum(N))/sum(N == 0);
%         N(N == 0) = perc;    
%         N = N/sum(N);
%         HMModel.DurationParameters.PNonParametric(k, :) = [zeros(1, HMModel.DurationParameters.dmin) N];
%     end
% end

%         dmin = min(dmink);
%         dmax = max(dmaxk);
%         for k = 1:K
%             %daux = cell2mat(dState(:,k)');
%             %daux = daux(daux >= HMModel.DurationParameters.dmin);
%             daux = dStateall{k};
%             if k == 1
%                 [f, ~] = ksdensity(daux, dmin:dmax);
%             else
%                 f = normpdf(dmin:dmax, 100, 50);        % Only works for Fs = 100 Hz!!!!!!!!!
%                 %f = ones(1, dmax - dmin + 1);
%             end
%             f = f/sum(f);
%             HMModel.DurationParameters.PNonParametric(k, :) = [zeros(1, dmin-1) f];
%         end


%%
%HMModel.DurationParameters.dmax = dmax;
%HMModel.DurationParameters.dmin = dmin;
HMModel.DurationParameters.flag = 0;

HMModel.StateParameters.A(:, :, 1) = A;
HMModel.StateParameters.A(:,:,2) = eye(K);
end

end