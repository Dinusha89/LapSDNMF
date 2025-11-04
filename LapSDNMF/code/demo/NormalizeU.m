% Normalization is performed in column wise since samples are represented
% using columns
function Matrix = NormalizeU(Matrix, viewv, layeri, typeOfNormalization)      
    if(typeOfNormalization == "None")
        return;
    else
        if(typeOfNormalization == "L1")
            Matrix{viewv,layeri} = normalize(Matrix{viewv,layeri}, 1, 'norm',1);
        elseif(typeOfNormalization == "L2")
            Matrix{viewv,layeri} = normalize(Matrix{viewv,layeri}, 1, 'norm',2);
        elseif(typeOfNormalization == "MinMax")
            Matrix{viewv,layeri} = normalize(Matrix{viewv,layeri}, 1, 'range');
        elseif(typeOfNormalization == "Std")
            Matrix{viewv,layeri} = zscore(Matrix{viewv,layeri}, 0, 1);
        end    
    end
end

