function z = resampleLabels(zUp, y, winLen, ovp)

loc = winLen:winLen-ovp:length(y);
z = ones(1, numel(y));
for i = 1:numel(loc)
    if zUp(i) == 2
        z(loc(i) - winLen + 1:loc(i)) = 2;
    end
end

end