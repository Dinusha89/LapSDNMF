function [U_final, Z_final, F_final, nIter_final, obj_all] = LapSDNMF_Multi(X, Y, layers, nCluster, S, W, options, nviews, U, Z, F)
    % where
    %   X
    % Notation:
    % X ... (mFea x nSmp) data matrix 
    %       mFea  ... number of dimensions 
    %       nSmp  ... number of samples
    % A/Y ... (nSmp x c)Label matrix of X
    
    % r...  r << min(mFea , nSmp) number of hidden factors/subspace dimensions
    % layers -> This is equivalent #layers and it's dimentions in deep architecture
    
    % nCluster ... number of classes
    % S ... (nSmp x nSmp)diagonal label matrix
    % W ... (nSmp x nSmp)weight matrix of the affinity graph 
    % U ... (mFea x r) base
    % Z ... (nCluster x r) auxiliary
    % F ... (nSmp x nCluster) membership
    
    % options ... Structure holding all settings
    %
    % You only need to provide the above four inputs.
    %
    % X = U*(FZ)'

    isPlottingRequired = 0;
    
    differror = options.error;
    maxIter = options.maxIter;
    %nRepeat = options.nRepeat;
    minIter = options.minIter - 1;
    typeOfNormalization = options.typeOfNormalization;
    typeOfInitialization = options.typeOfInitialization;
    dataset = extractBefore(options.dataset,'.mat');

    if ~isempty(maxIter) && maxIter < minIter
        minIter = maxIter;
    end
    meanFitRatio = options.meanFitRatio;
    
    % alpha in code = Lamda in code 
    alpha = options.alpha;
    nLayers = length(layers);
    nRepeat = 1;
    
    %Norm and NormF are always set to these two values according to LR
    Norm = 2;
    NormF = 1;
    
    obj_all=[];
    
    mFea = zeros(numel(X),1);
    for i = 1:nviews
        [mFea(i),nSmp]=size(X{i,1});
    end
    
    if alpha > 0
        
        DCol = cell(numel(X),1);
        D = cell(numel(X),1);
        L = cell(numel(X),1);
    
        for i = 1:nviews
            %W{i,1} = alpha * W{i,1};
       
            DCol{i,1} = full(sum(W{i,1},2));
            D{i,1} = spdiags(DCol{i,1},0,nSmp,nSmp);
            L{i,1} = D{i,1} - W{i,1};
    
        end
    else
        L = [];
    end
    
    selectInit = 1;    
    if isempty(U)
        [U, Z, F] = InitializeMatrices(X, mFea, nSmp, nCluster, layers, nviews, typeOfInitialization);
    else
        nRepeat = 1;  
    end
    
    [U, Z, F] = NormalizeMatrices(U, Z, F, nviews, nLayers, NormF, Norm);
    
    % nRepeat = 10 be default if we don't manually set it
    if nRepeat == 1
        selectInit = 0;
        minIter = 0;
        if isempty(maxIter)
            objhistory = CalculateObj(X, U, Z, F, S, Y, L, nviews, alpha, nLayers);
            meanFit = objhistory*10;     
        else
            if isfield(options,'Converge') && options.Converge
                objhistory = CalculateObj(X, U, Z, F, S, Y, L, nviews, alpha, nLayers);
            end
        end
    else
        if isfield(options,'Converge') && options.Converge
            error('Not implemented!');
        end
    end
    
    
    tryNo = 0;
    nIter = 0;
    nCal=0;

    if(isPlottingRequired == 1)
        ErrorToPlot = zeros(maxIter);
    end
    
    while tryNo < nRepeat   
        tryNo = tryNo+1;
        maxErr = 1;
        while maxErr > differror

            %fprintf('maxErr = %2.4f / differror = %2.4f \n',maxErr, differror);

            for viewv = 1:nviews
                for layeri = 1:nLayers

                    % ===================== update U ========================

                    if layeri == 1
                        UUp = X{viewv,1} * F{viewv, 1} * Z{viewv, layeri};
                        UDown = U{viewv, layeri} * transpose(Z{viewv, layeri}) * transpose(F{viewv, 1}) * F{viewv, 1} * Z{viewv, layeri};
                    else
                        ThetaMinusTranspose = CalculateThetaTranspose(U, viewv, layeri-1);
                        ThetaMinus = CalculateTheta(U, viewv, layeri-1);
                    
                        UUp = ThetaMinusTranspose * X{viewv,1} * F{viewv, 1} * Z{viewv, layeri};
                        UDown = ThetaMinusTranspose * ThetaMinus * U{viewv, layeri} * transpose(Z{viewv, layeri}) * transpose(F{viewv, 1}) * F{viewv, 1} * Z{viewv, layeri};
                    end

                    U{viewv, layeri} = U{viewv, layeri}.* sqrt(UUp ./max(UDown, 1e-10));
                    U = NormalizeU(U, viewv, layeri, typeOfNormalization);
                    
                   % ===================== update Z ========================

                   ThetaTranspose = CalculateThetaTranspose(U, viewv, layeri);
                   Theta = CalculateTheta(U, viewv, layeri);

                   ZUp = transpose(F{viewv, 1}) * transpose(X{viewv,1}) * Theta;
                   if alpha ~=0
                       ZDown = (transpose(F{viewv, 1}) * F{viewv, 1} * Z{viewv, layeri} * ThetaTranspose * Theta) + alpha*(transpose(F{viewv, 1}) * L{viewv, 1} * F{viewv, 1} * Z{viewv, layeri});
                   else 
                        ZDown = (transpose(F{viewv, 1}) * F{viewv, 1} * Z{viewv, layeri} * ThetaTranspose * Theta);
                   end
                   
                   Z{viewv, layeri} = Z{viewv, layeri}.* sqrt(ZUp ./max(ZDown, 1e-10));
                   Z = NormalizeZ(Z, viewv, layeri, typeOfNormalization);
                    
                   % ===================== update F ========================   
                   
                   % This Sigma contains the multiplication of Z and transpose(Z) terms
                   SigmaZ = CalculateSigmaZ(Z, viewv, nLayers);
                   
                   FUp = transpose(X{viewv,1}) * Theta * transpose(Z{viewv, layeri}) + 2*S*Y;
                   if alpha ~=0
                        FDown = F{viewv, 1} * Z{viewv, layeri} * ThetaTranspose * Theta * transpose(Z{viewv, layeri}) + alpha*(L{viewv, 1} * F{viewv, 1} * SigmaZ) + S*F{viewv, 1};
                   else 
                        FDown = F{viewv, 1} * Z{viewv, layeri} * ThetaTranspose * Theta * transpose(Z{viewv, layeri}) + S*F{viewv, 1};
                   end
                   
                   F{viewv, 1} = F{viewv, 1} .* sqrt(FUp ./max(FDown, 1e-10));
                   F = NormalizeF(F, viewv, typeOfNormalization);

                end
            end
            
           % =========================End of Update Rules================================ 

           nIter = nIter + 1;
           if nCal < maxIter
                if nIter <= maxIter
                    obj = CalculateObj(X, U, Z, F, S, Y, L, nviews, alpha, nLayers);

                    obj_all =[obj_all obj];
                    
                    if(isPlottingRequired == 1)
                        if(nIter>0)
                            ErrorToPlot(nIter) = obj;
                        end
                        fprintf('Iteration = %d / Objective Function Value = %f \n',nIter, obj);
                    end

                end
                nCal = nCal + 1;
           end
           if nIter > minIter
                if selectInit
                    objhistory = CalculateObj(X, U, Z, F, S, Y, L, nviews, alpha, nLayers);
                    maxErr = 0;
                else
                    if isempty(maxIter)
                        newobj = CalculateObj(X, U, Z, F, S, Y, L, nviews, alpha, nLayers);
                        objhistory = [objhistory newobj];
                        meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                        maxErr = (meanFit-newobj)/meanFit;
                    else
                        if isfield(options,'Converge') && options.Converge
                            newobj = CalculateObj(X, U, Z, F, S, Y, L, nviews, alpha, nLayers);
                            objhistory = [objhistory newobj];
                        end
                        maxErr = 1;
                        if nIter >= maxIter
                            % stopping condition when max iteration is reached
                            maxErr = 0;
                            if isfield(options,'Converge') && options.Converge
                            else
                                objhistory = 0;
                            end
                        end
                    end
                end
           end
        end
        
        if(isPlottingRequired == 1)
            plot(ErrorToPlot);
            title(dataset);
            xlabel('Number of Iterations'); 
            ylabel('Objective Value');
            
            
            relativePath = '../../Figures/ConvergenceCurves/';
            fileName = strcat(dataset, 'ConvergenceCurve.png');
            currentScriptDir = fileparts(mfilename('fullpath'));
            filePathAndName = fullfile(currentScriptDir, relativePath, fileName);
            exportgraphics(gcf,filePathAndName,'BackgroundColor','none','Resolution',1080);
            
        end
        
        if tryNo == 1
            U_final = U;
            Z_final = Z;
            F_final = F;  
            nIter_final = nIter;
            objhistory_final = objhistory;
        else
           if objhistory(end) < objhistory_final(end)
               U_final = U;
               Z_final = Z;
               F_final = F;  
               nIter_final = nIter;
               objhistory_final = objhistory;
           end
        end
    
        if selectInit
            if tryNo < nRepeat
                %re-start
                [U, Z, F] = InitializeMatrices(X, mFea, nSmp, nCluster, layers, nviews, typeOfInitialization);
                [U, Z, F] = NormalizeMatrices(U, Z, F, nviews, nLayers, NormF, Norm);
            else
                tryNo = tryNo - 1;
                nIter = minIter+1;
                selectInit = 0;
                U = U_final;
                Z = Z_final;
                F = F_final;
                objhistory = objhistory_final;
                meanFit = objhistory * 10;
            end
        end
    end
    
    [U_final, Z_final, F_final] = NormalizeMatrices(U_final, Z_final, F_final, nviews, nLayers, NormF, Norm);

end

%==========================================================================
function [obj] = CalculateObj(X, U, Z, F, S, Y, L, nviews, alpha, nLayers)

    part_one = 0;
    for viewv = 1:nviews
        
        part_one = part_one + norm(X{viewv,1} - (CalculateTheta(U, viewv, nLayers)*(transpose(Z{viewv, nLayers}))*(transpose(F{viewv, 1}))), 'fro');
    end

    part_two = 0;
    if alpha ~=0
        for viewv = 1:nviews
            for layeri = 1:nLayers

                part_two = part_two + trace(transpose(Z{viewv, layeri})*transpose(F{viewv, 1})*L{viewv, 1}*F{viewv, 1}*Z{viewv, layeri}) + trace((transpose(F{viewv, 1}) - transpose(Y))*S*(F{viewv, 1} - Y));
            end
        end
        part_two = alpha*part_two;
    end 
    
    obj = part_one + part_two;
end

function Theta = CalculateTheta(U, viewv, nLayers)
    Theta = 0;
    
    for layeri = 1:nLayers
        if layeri == 1
            Theta = U{viewv, layeri};
        else
            Theta = Theta * U{viewv, layeri};
        end
    end
end

function ThetaTranspose = CalculateThetaTranspose(U, viewv, nLayers)
    ThetaTranspose = 0;
    
    for layeri = nLayers:-1:1
        if layeri == nLayers
            ThetaTranspose = transpose(U{viewv, layeri});
        else
            ThetaTranspose = ThetaTranspose * transpose(U{viewv, layeri});
        end
    end
end

function SigmaZ = CalculateSigmaZ(Z, viewv, nLayers)
    SigmaZ = 0;
    
    for layeri = 1:nLayers
        SigmaZ = SigmaZ + Z{viewv, layeri} * transpose(Z{viewv, layeri});
    end
end