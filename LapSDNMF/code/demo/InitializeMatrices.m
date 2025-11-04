function [U, Z, F] = InitializeMatrices(X, mFea, nSmp, nCluster, layers, nviews, typeOfInitialization)

    nLayers = length(layers);
    
    U = cell(nviews, nLayers);
    Z = cell(nviews, nLayers);
    F = cell(nviews, 1);
    
    if(typeOfInitialization == "Random")
        for viewv = 1:nviews
            F{viewv, 1} = abs(rand(nSmp, nCluster));
            for layeri = 1:nLayers
    
                if layeri ==1
                    U{viewv, layeri} = abs(rand(mFea(viewv), layers(layeri)));
                else
                    U{viewv, layeri} = abs(rand(layers(layeri-1), layers(layeri)));
                end 
    
                Z{viewv, layeri} = abs(rand(nCluster, layers(layeri)));
            end
        end
    elseif(typeOfInitialization == "LiteKmeans")
            
            label = cell(nviews,1);

            for viewv = 1:nviews
                for layeri = 1:nLayers

                    if layeri ==1
                        U{viewv,layeri} = ones(mFea(viewv), layers(layeri));
                    else
                        U{viewv,layeri} = ones(layers(layeri-1), layers(layeri));
                    end

                    Z{viewv, layeri} = ones(nCluster, layers(layeri));
                    
                end
                label{viewv} = litekmeans(transpose(X{viewv, 1}), nCluster);
                
                for i = 1:nCluster
                    F{viewv,1}(label{viewv} == i, i) = 1;
                end
    
                F{viewv,1} = (F{viewv,1}+0.2);
            end
    end

end

