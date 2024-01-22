%****************************************************
%       Copyright (C) 2008 Nasser Abbasi
%  Free to use and modify for academic and research only
%***************************************************
function [the_intepolation,the_filter] = get_iradon_options(client_handles)

k = get(client_handles.iradonFilterTag,'Value');
switch k
    case 1
        the_filter = 'Ram-Lak';
    case 2
        the_filter = 'Shepp-Logan';
    case 3
        the_filter =  'Cosine' ;
    case 4
        the_filter = 'Hamming' ;
    case 5
        the_filter = 'Hann' ;
    case 6
        the_filter = 'None';
end

k = get(client_handles.iradonInterpTag,'Value');
switch k
    case 1
        the_intepolation = 'nearest';
    case 2
        the_intepolation = 'linear';
    case 3
        the_intepolation =  'spline' ;
    case 4
        the_intepolation= 'pchip' ;
    case 5
        the_intepolation = 'cubic' ;
end
