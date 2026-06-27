function colors = parulaHighContrast(nColors)
    cmap = parula(64);
    idx = round(linspace(1, size(cmap,1), nColors));
    colors = cmap(idx, :);
end
