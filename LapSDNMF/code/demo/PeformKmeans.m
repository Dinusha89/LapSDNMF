% Run k-means n times and report means and standard deviations of the
% performance measures.
%
% -------------------------------------------------------
% Input:
%       X:  data matrix (rows are samples)
%       nCluster:  number of clusters
%       gnd:  ground truth (#samples x 1)
%       semiSplit: labeled sample portion (#samples x 1)
%     
%
% Output:
%       purMean/purStd: Purity (mean + stdev)
%       accMean/accStd: clustering accuracy for unlabeled data portion (mean + stdev)
%       nmiMean/nmiStd: normalized mutual information (mean + stdev)
%       fscoreMean/fscoreStd: F1 measure (mean + stdev)

function [purMean, purStd, accMean, accStd, nmiMean, nmiStd, fscoreMean, fscoreStd] = PeformKmeans(X, nCluster, gnd, semiSplit)

    %max_iter = 1000; % Maximum number of iterations for KMeans
    replic = 20; % Number of replications for KMeans
    
    if (min(gnd)==0)
        gnd = gnd+1;
    end
    
    warning('off');
    
    pur = zeros(1, replic);
    acc = zeros(1, replic);
    nmi = zeros(1, replic);
    fscore = zeros(1, replic);
    
    for i=1:replic
        label = kmeans(X, nCluster);
    
        pur(i) = Purity(gnd', label');
        label = bestMap(gnd,label);
        acc(i) = length(find(gnd(~semiSplit) == label(~semiSplit)))/length(gnd(~semiSplit));
        nmi(i) = MutualInfo(gnd(~semiSplit),label(~semiSplit));
        fscore(i) = Fscore(gnd',label');
    end
    
    % Calculate mean and STD for selected measures and round them for four decimal places
    purMean = round(mean(pur),4);
    purStd = round(std(pur),4);
    accMean = round(mean(acc),4);
    accStd = round(std(acc),4);
    nmiMean = round(mean(nmi),4);
    nmiStd = round(std(nmi),4);
    fscoreMean = round(mean(fscore),4);
    fscoreStd = round(std(fscore),4);

end