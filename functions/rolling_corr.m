function roll = rolling_corr(r1, r2, T)
    y = zscore(r2);
    n = size(y, 1);

    if (n < T)
        roll = [];
    else
        x = zscore(r1);
        x2 = x .^ 2;
        y2 = y .^ 2;
        xy = x .* y;
        A = 1;
        B = ones(1,T);
        stdx = sqrt((filter(B, A, x2) - (filter(B, A, x) .^ 2) ...
               * (1 / T)) / (T - 1));
        stdy = sqrt((filter(B, A, y2) - (filter(B, A, y) .^ 2) ...
               * (1 / T)) / (T - 1));
        roll = (filter(B, A, xy) - ...
                filter(B, A, x) .* filter(B, A, y) / T) ...
               ./ ((T - 1) * stdx .* stdy);
        roll = roll(T : end);
    end
end