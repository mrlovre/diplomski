function [x, y] = argmin(M)
    switch length(size(M))
        case 1
            [~, x] = min(M);
        case 2
            [~, i] = min(tovector(M));
            [x, y] = ind2sub(size(M), i);
        case 3
            dim = size(M);
            [~, i] = min(reshape(M, [dim(1), prod(dim(2 : 3))]), [], 2);
            [x, y] = ind2sub(dim(2 : 3), i);
        otherwise
            error('Works only with 1, 2 or 3 dimensions.');
    end
end