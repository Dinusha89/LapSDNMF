function [U_final, Z_final, F_final, nIter_final, obj_all] = LapSDNMF(X, Y, layers, nCluster, S, W, options, nviews, U, Z, F)

    % where
    %   X
    % Notation:
    % X ... (mFea x nSmp) data matrix 
    %       mFea  ... number of dimensions 
    %       nSmp  ... number of samples
    % A/Y ... (nSmp x c)Label matrix of X
    
    % r...  r << min(mFea , nSmp)number of hidden factors/subspace dimensions /
    % layers -> This is equivalent #layers and it's dimentions in deep architecture
    
    % nCluster ... number of classes
    % S ... (nSmp x nSmp)diagonal label matrix
    % W ... (nSmp x nSmp)weight matrix of the affinity graph 
    % U ... (mFea x r) base metric
    % Z ... (nCluster x r)  auxiliary metric
    % F ... (nSmp x nCluster) Prediction membership matrix
    
    
    % options ... Structure holding all settings
    %
    % You only need to provide the above four inputs.
    %
    % X = U * (FZ)'
    
    for i = 1:nviews
        if min(min(X{i,1})) < 0
            error('Input should be nonnegative!');
        end
    
        %This transpose was taken since it was skipped at the call point of this method
        X{i,1} = X{i,1}';
    end
    
    if ~isfield(options,'error')
        options.error = 1e-5;
    end
    if ~isfield(options, 'maxIter')
        options.maxIter = [];
    end
    
    if ~isfield(options,'nRepeat')
        options.nRepeat = 10;
    end
    
    if ~isfield(options,'minIter')
        options.minIter = 30;
    end
    
    if ~isfield(options,'meanFitRatio')
        options.meanFitRatio = 0.1;
    end
    
    if ~isfield(options,'alpha')
        options.alpha = 10;
    end
    
    if isfield(options,'alpha_nSmp') && options.alpha_nSmp
        options.alpha = options.alpha*nSmp;    
    end
    
    if ~isfield(options,'Optimization')
        options.Optimization = 'Multiplicative';
    end
    
    if ~isfield(options,'typeOfNormalization')
        options.typeOfNormalization = 'None';
    end
    
    if ~exist('U','var')
        U = [];
        Z = [];
        F = [];
    end
    
    switch lower(options.Optimization)
        case {lower('Multiplicative')} 
            [U_final, Z_final, F_final, nIter_final, obj_all] = LapSDNMF_Multi(X, Y, layers, nCluster, S, W, options, nviews, U, Z, F);
        otherwise
            error('optimization method does not exist!');
    end

end


    
        