%****************************************************
%       Copyright (C) 2008 Nasser Abbasi
%  Free to use and modify for academic and research only
%***************************************************

function make3dspectrum(axesHandle,F,clip,c)

axes(axesHandle);
L      = size(F,1);
[i,j]  = find(F>(clip/100)*(max(max(F))));
d      = F;
d(sub2ind(size(F),i,j)) = 0;
r      = round(.15*L);
extent = r:L-r;
mesh(extent,extent,c*log(1+d(extent,extent))); 
axis('tight');
shading interp;
xlabel('u');
ylabel('v');
view(-45,50);
end