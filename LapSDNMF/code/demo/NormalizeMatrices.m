function [U, Z, F] = NormalizeMatrices(U, Z, F, nviews, nLayers, NormF, Norm)
    
    for viewv = 1:nviews
        for layeri = 1:nLayers

            if Norm == 2         
                if NormF 
                    norms = max(1e-15,sqrt(sum(F{viewv, 1}.^2,1)))'; 
                    F{viewv, 1} = F{viewv, 1} * spdiags(norms.^-1,0,size(F{viewv, 1},2),size(F{viewv, 1},2));
                    normsu = max(1e-15,sqrt(sum(U{viewv, layeri}.^2,1)))';
                    U{viewv, layeri} = U{viewv, layeri} * spdiags(normsu.^-1,0,size(U{viewv, layeri},2),size(U{viewv, layeri},2));
                    Z{viewv, layeri} = spdiags(sqrt(norms),0,size(F{viewv, 1},2),size(F{viewv, 1},2)) * Z{viewv, layeri} * spdiags(sqrt(normsu),0,size(Z{viewv, layeri},2),size(Z{viewv, layeri},2));
                end
            else
                if NormF
                    norms = max(1e-15,sum(abs(F{viewv, 1}),1))';
                    F{viewv, 1} = F{viewv, 1} * spdiags(norms.^-1,0,size(F{viewv, 1},2),size(F{viewv, 1},2));
                    U{viewv, layeri} = U{viewv, layeri} * spdiags(sqrt(norms),0,size(U{viewv, layeri},2),size(U{viewv, layeri},2));
                    Z{viewv, layeri} = Z{viewv, layeri} * spdiags(sqrt(norms),0,size(Z{viewv, layeri},2),size(Z{viewv, layeri},2));
                end
            end

        end
    end

end

