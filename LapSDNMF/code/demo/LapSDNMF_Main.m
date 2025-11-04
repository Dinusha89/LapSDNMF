% demo label percent with all datasets for acc, NMI, purity, F-score

function LapSDNMF_Main(dataset)
    
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
    addpath(genpath(pwd));
    rng("default");

    %fileName = strcat('LapSDNMF_LogFor_', extractBefore(dataset,'.mat'),'.txt');
    %diary (fileName);
    
    % choose a dataset
    %datasets = ['BBC.mat', 'BBCSport.mat', 'BRCA.mat', 'Caltech7.mat', 'Caltech20.mat', 'HandWritten.mat', 'Leaves100.mat', 'ORL.mat', 'ReutersMinMax.mat', 'ROSMVP.mat', 'Threesources.mat', 'Yale.mat'];

    relativeDataPath = '../../.././datasets/';
    currentScriptDir = fileparts(mfilename('fullpath'));
    dataPathAndName = fullfile(currentScriptDir, relativeDataPath, dataset);
    load(dataPathAndName);
    
    %fea = samplesxfeatures, gnd = samplesx1, nviews (These comes from the actual dataset)
    correctFea = fea;
    correctGnd = gnd;
    
    %Starting from here
    nClass  = length(unique(correctGnd));   %nClass
    
    % Get the unique classes
    classes = unique(correctGnd);
    
    % Count the occurrences of each class
    classCounts = histcounts(correctGnd', [classes', max(classes)+1]);
    
    % min value is set as nEach
    nEach = min(classCounts);
    
    %normalization the feature (sample wise normalization)
    for viewv = 1:nviews
        correctFea{viewv,1} = NormalizeFea(correctFea{viewv,1});
    end
    
    %nDim = #features in each view
    nDim = zeros(numel(correctFea),1);
    for viewv = 1:nviews
        nDim(viewv)    = size(correctFea{viewv,1},2);
    end
    %--------------------------------------------------------------------------
    
    nCluster = nClass;
    labelPercentage = 0.3;
    lamda = 0.001;
    heatKernel = 0.1;
    %Neighbourhood size k in KNN
    neighbourhoodSize = 11;
    
    %Equivalent rank r
    layers = [100 50];
    % Eligible weight modes -> Binary, HeatKernel, Cosine
    weightMode = 'HeatKernel';
    % Eligible normalization types -> L1, L2, MinMax, Std 
    typeOfNormalization = 'L2';
    % Eligible initialization types -> Random, LiteKmeans
    typeOfInitialization = 'Random';
    
    maxIter = 1000;

    isParaTrainReq = 0;
    totalMetrics = 13;
    
    if isParaTrainReq ==0 
        DoClustering(nEach, nCluster, correctFea, nviews, nDim, nClass, correctGnd, labelPercentage, weightMode, neighbourhoodSize, maxIter, dataset, lamda, heatKernel, typeOfNormalization, typeOfInitialization, layers, isParaTrainReq);
    else
       
        %labelPercentageArray = [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
        labelPercentageArray = [0.3];
        lamdaArray = [0.001 0.01 0.1 1 10 100 1000];
        layersArray = [100 50 0 0; 150 50 0 0; 150 100 50 0; 300 200 100 0; 300 200 125 0; 100 50 25 0; 200 150 100 50];
        neighbourhoodSizeArray = [3 5 7 9 11 13];
        heatKernelArray = [0.001 0.01 0.1 1 10 100 1000];
        
        expRawCount = getTotalRawCount(labelPercentageArray, lamdaArray, layersArray, neighbourhoodSizeArray, heatKernelArray);
        currentCount = 0;
        FinalResults = cell(expRawCount,totalMetrics);

        for labelPercentageArrayIndex = 1 : length(labelPercentageArray)
            labelPercentage = labelPercentageArray(labelPercentageArrayIndex);

            for lamdaArrayIndex = 1 : length(lamdaArray)
                lamda = lamdaArray(lamdaArrayIndex);

                for layersArrayIndex = 1 : size(layersArray,1)

                    layer1 = layersArray(layersArrayIndex,1);
                    layer2 = layersArray(layersArrayIndex,2);
                    layer3 = layersArray(layersArrayIndex,3);
                    layer4 = layersArray(layersArrayIndex,4);
                    
                    if(layer2 ==0 && layer3 ==0 && layer4 ==0)
                        layers = [layer1];
                    elseif(layer3 ==0 && layer4 ==0)
                        layers = [layer1 layer2];
                    elseif(layer4 ==0)
                        layers = [layer1 layer2 layer3];
                    else
                        layers = [layer1 layer2 layer3 layer4];
                    end
                    
                    for neighbourhoodSizeArrayIndex = 1 : length(neighbourhoodSizeArray)
                        neighbourhoodSize = neighbourhoodSizeArray(neighbourhoodSizeArrayIndex);

                        for heatKernelArrayIndex = 1 : length(heatKernelArray)
                            heatKernel = heatKernelArray(heatKernelArrayIndex);
                        
                            currentCount = currentCount + 1;
                            [FinalResults] = DoClustering(nEach, nCluster, correctFea, nviews, nDim, nClass, correctGnd, labelPercentage, weightMode, neighbourhoodSize, maxIter, dataset, lamda, heatKernel, typeOfNormalization, typeOfInitialization, layers, isParaTrainReq, currentCount, FinalResults);
                        end
                    end
                end
            end
        end
    end
    %diary off;
end

function [FinalResults] = DoClustering(nEach, nCluster, correctFea, nviews, nDim, nClass, correctGnd, labelPercentage, weightMode, neighbourhoodSize, maxIter, dataset, lamda, heatKernel, typeOfNormalization, typeOfInitialization, layers, isParaTrainReq, currentCount, FinalResults)
    
    nLayers = length(layers);    
    nSample  = nEach*nCluster;
    Samples = cell(numel(correctFea),1);
    
    for viewv = 1:nviews
        Samples{viewv,1} = zeros(nSample,nDim(viewv));
    end
    labels  = zeros(nSample,1);
    
    shuffleClasses = randperm(nClass);
    sampleEach = cell(numel(correctFea),1);

    for viewv = 1:nviews
        index  = 0;
        for class = 1:nCluster 
            idx = find(correctGnd == shuffleClasses(class));
            sampleEach{viewv,1} = correctFea{viewv,1}(idx(1:nEach),:); 
            Samples{viewv,1}(index+1:index+nEach,:) = sampleEach{viewv,1};
            %since labels are same accross multiple views this should run only for first view
            if(viewv==1)
                labels(index+1:index+nEach,:)  = class; 
            end
            index = index + nEach;
        end
    end
    
    %---------------------------------------------------------------------
    feaSet = Samples;
    gndSet = labels;
    semiSplit = false(size(gndSet));
    
    for class = 1:nCluster  %
        idx = find(gndSet == class);
        shuffleIndexes = randperm(length(idx));
        nSmpLabelsRequired = floor((labelPercentage)*length(idx));
        semiSplit(idx(shuffleIndexes(1:nSmpLabelsRequired))) = true;
    end
    
    %since nfeaSet taking size of number of samples this remains same for all views
    nfeaSet = size(feaSet{1,1},1);
    %nSmpLabeled = sum(semiSplit);
    
    % shuffle the data sets and lables
    shuffleIndexes = randperm(nfeaSet); % 1 * 500k
    
    for viewv = 1:nviews
        feaSet{viewv,1} = feaSet{viewv,1}(shuffleIndexes,:);
        % setting up negative values to zero if any - applicable only for 'Yale.mat'
        if(dataset == "Yale.mat")
            feaSet{viewv,1} = max(feaSet{viewv,1},0);
        end
    end
    
    gndSet = gndSet(shuffleIndexes);
    semiSplit = semiSplit(shuffleIndexes);  % 500k * 1 logical
    
    % constructing the similarity diagnal matrix based on labled data
    S = diag(semiSplit); % semiSplit(500k * 1 logical) diagonal metirx 500k * 500k
    
    % constructing the label constraint matrix for LpCNMF and CNMF
    E = eye(nCluster); % nCluster * nCluster diagonal metirx
    A_mid = E(:,gndSet(semiSplit)); % nCluster * (500k*labelPercentage(0.3))
    
    Y = zeros(nCluster,nfeaSet); % nCluster * 500k
    Y(:,semiSplit) = A_mid;  % column with label turn to A_mid

    %--------------------------------------------------------------------------------------
    %Clustering by Label Propagation Constrained Deep Multi-view Non-negative Matrix Factorization
    options = [];
    options.WeightMode = weightMode;
    options.NeighborMode = 'KNN';
    options.k = neighbourhoodSize;   
    options.t = heatKernel;
    options.maxIter = maxIter;
    options.dataset = dataset;
    
    if(lamda>0)
        W = cell(numel(correctFea),1);
        for viewv = 1:nviews
            %W{viewv,1} = constructW(feaSet{viewv,1},options);
            W{viewv,1} = full(constructW(feaSet{viewv,1},options));
        end
    else
        W = [];
    end
    
    %This is the lamda in objective function
    options.alpha = lamda;
    options.typeOfNormalization = typeOfNormalization;
    options.typeOfInitialization = typeOfInitialization;

    layerDimension = '';
    for i = 1:length(layers)
        layerDimension = [layerDimension, num2str(layers(i)), ' '];
    end
    layerDimension = strtrim(layerDimension);
    layerDimension = ['[', layerDimension, ']'];
    
    fprintf('Lamda = %4.3f, KNN - Neighbourhood Size = %d, Label Presentage = %1.2f, Layer Dimension = %s, heatKernel = %4.3f, Selected Sample Size = %d \n', lamda, neighbourhoodSize, labelPercentage, layerDimension, heatKernel, nfeaSet);
    
    % LapSDNMF method
    [~, Zest_lpcnmf, F, nIter_final, obj_all] = LapSDNMF(feaSet, Y', layers, nCluster, S, W, options, nviews);
    
    FZ_com = zeros(nfeaSet,layers(nLayers));
    FZ = cell(numel(correctFea),1);
    for viewv=1:nviews
        FZ{viewv,1} = F{viewv,1}*Zest_lpcnmf{viewv,nLayers};
        FZ_com = FZ_com + 1/nviews*FZ{viewv,1};
    end

    [purMean, purStd, accMean, accStd, nmiMean, nmiStd, fscoreMean, fscoreStd] = PeformKmeans(FZ_com, nCluster, gndSet, semiSplit);
    
    if isParaTrainReq ==0

        fprintf('NMI = %f and STD = %f \n',nmiMean, nmiStd);
        fprintf('Accuracy = %f and STD = %f \n',accMean, accStd);
        fprintf('F-score = %f and STD = %f \n',fscoreMean, fscoreStd);
        fprintf('Purity = %f and STD = %f \n',purMean, purStd);

    else
        FinalResults{currentCount,1} = nmiMean;
        FinalResults{currentCount,2} = accMean;
        FinalResults{currentCount,3} = fscoreMean;
        FinalResults{currentCount,4} = purMean;

        FinalResults{currentCount,5} = labelPercentage;
        FinalResults{currentCount,6} = lamda;
        FinalResults{currentCount,7} = layers;
        FinalResults{currentCount,8} = neighbourhoodSize;
        FinalResults{currentCount,9} = heatKernel;

        FinalResults{currentCount,10} = nmiStd;
        FinalResults{currentCount,11} = accStd;
        FinalResults{currentCount,12} = fscoreStd;
        FinalResults{currentCount,13} = purStd;
        
        relativePath = '../../Results/';
        resultFileName = strcat('FinalResult', dataset);
        currentScriptDir = fileparts(mfilename('fullpath'));
        filePathAndName = fullfile(currentScriptDir, relativePath, resultFileName);
        save(filePathAndName,'FinalResults');
    end
end

function totalCount = getTotalRawCount(labelPercentageArray, lamdaArray, layersArray, neighbourhoodSizeArray, heatKernelArray)
    
    totalCount = 0;
    for labelPercentageArrayIndex = 1 : length(labelPercentageArray)
        for lamdaArrayIndex = 1 : length(lamdaArray)
            for layersArrayIndex = 1 : size(layersArray,1)
                for neighbourhoodSizeArrayIndex = 1 : length(neighbourhoodSizeArray)
                    for heatKernelArrayIndex = 1 : length(heatKernelArray)
                        totalCount = totalCount + 1;
                    end
                end
            end
        end
    end
end




 

