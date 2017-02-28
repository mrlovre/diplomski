begin = 1700;
finish = 2400;
deltas = sp500_logprice(begin - T : finish, 190) - sp500_logprice(begin - T : finish, 196);
averages = movmean(deltas, T, 'Endpoint', 'discard');
stddevs = movstd(deltas, T, 'Endpoint', 'discard');
d = 2;

decisions = zeros(finish - begin + 1, 1);
for t = 1 : finish - begin + 1
    if deltas(i + T) > averages(i) + d * stddevs(i)
        decisions(i) = 1;
    elseif deltas(i + T) < averages(i) - d * stddevs(i)
        decisions(i) = -1;
    end
end
