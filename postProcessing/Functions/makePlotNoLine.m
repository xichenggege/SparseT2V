function ch = makePlotNoLine(X,Y,Field, Mfac,MM)
vv = -1:0.1:1;
vv(11)=[];
vv(1) = -10;
vv(end) = 10;
if nargin == 4
    MM = max(abs(Field(:)))*Mfac;
end
[cv,ch] = contourf(X,Y,Field,MM*vv,'LineColor','none'); shading flat
caxis([-MM MM]);
set(gca,'fontsize',14);
hold on
cmap = redblue(20);
cmap(10:11,:)=1;
colormap(cmap);
axis equal
% axis off
end
