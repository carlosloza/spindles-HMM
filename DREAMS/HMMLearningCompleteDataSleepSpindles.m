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
for k = 1:K
    idx = find(labelsall == k);
    Ytil = yauxall(idx)';
    Xtil = Yp(idx, :)';
    if HMModel.robustIni
        mdl = fitlm(Xtil', Ytil', 'Intercept', false, 'RobustOpts', 'welsch');
    else
        mdl = fitlm(Xtil', Ytil', 'Intercept', false);
    end    
    HMModel.ObsParameters.meanParameters(k).Coefficients = table2array(mdl.Coefficients(:,1));
    err = Xtil'*HMModel.ObsParameters.meanParameters(k).Coefficients - Ytil;
    % Residual model, i.e. errors   
    pd = fitdist(err, 'tLocationScale');
    HMModel.ObsParameters.sigma(k) = pd.sigma;
    HMModel.ObsParameters.nu(k) = pd.nu;
end
%% Parameters of hidden markov chain
HMModel.StateParameters.pi = [1 0]';
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

HMModel.DurationParameters.flag = 0;
HMModel.StateParameters.A(:, :, 1) = A;
HMModel.StateParameters.A(:,:,2) = eye(K);


end