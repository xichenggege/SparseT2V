function [disc_] = discretizeBC(X,disc_edges,nmse)
%Discretize boundary conditions vs errors

disc_       = [];
% discretize erros into edges to analyze the error source
disc_.X     = X;      % data to be discretized
disc_.edges = disc_edges;  % edge for discretization
[disc_.Y,disc_.E] = discretize(disc_.X,disc_.edges); % index of each group/edge center

for i = 1:numel(disc_.edges)
    disc_.edgeIndex    = find(disc_.Y==i);
    disc_.nmse_mean(i) = mean(nmse(disc_.edgeIndex));
    if isempty(disc_.edgeIndex)
        disc_.nmse_min(i)  = nan;
        disc_.nmse_max(i)  = nan;
    else
        disc_.nmse_min(i)  = min(nmse(disc_.edgeIndex));
        disc_.nmse_max(i)  = max(nmse(disc_.edgeIndex));
    end
     disc_.nmse_std(i) = std(nmse(disc_.edgeIndex));
end
