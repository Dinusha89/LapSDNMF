% Normalization is performed in row wise since samples are represented
% using rows
function Matrix = NormalizeF(Matrix, viewv, typeOfNormalization)      
    if(typeOfNormalization == "None")
        return;
    else 
        if(typeOfNormalization == "L1")
            Matrix{viewv,1} = normalize(Matrix{viewv,1}, 2, 'norm',1);
        elseif(typeOfNormalization == "L2")
            Matrix{viewv,1} = normalize(Matrix{viewv,1}, 2, 'norm',2);
        elseif(typeOfNormalization == "MinMax")
            Matrix{viewv,1} = normalize(Matrix{viewv,1}, 2, 'range');
        elseif(typeOfNormalization == "Std")
            Matrix{viewv,1} = zscore(Matrix{viewv,1}, 0, 2);
        end
    end
end

