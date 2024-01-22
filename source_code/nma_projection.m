function varargout = nma_projection(varargin)
%
%  illustrate central slice theorem and reconstruction
%  of 2D images from backprojections using inverse radon transform
%
%  work related to Math 597 project, summer 2008
%  by Nasser Abbasi 6/2/08
%

% NMA_PROJECTION M-file for nma_projection.fig
%      NMA_PROJECTION, by itself, creates a new NMA_PROJECTION or raises the existing
%      singleton*.
%
%      H = NMA_PROJECTION returns the handle to a new NMA_PROJECTION or the handle to
%      the existing singleton*.
%
%      NMA_PROJECTION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NMA_PROJECTION.M with the given input arguments.
%
%      NMA_PROJECTION('Property','Value',...) creates a new NMA_PROJECTION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before nma_projection_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to nma_projection_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help nma_projection

% Last Modified by GUIDE v2.5 14-Nov-2008 14:04:41

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @nma_projection_OpeningFcn, ...
                   'gui_OutputFcn',  @nma_projection_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
end

% --- Executes just before nma_projection is made visible.
function nma_projection_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to nma_projection (see VARARGIN)

% Choose default command line output for nma_projection
handles.output = hObject;

set(handles.groupBtnTag,'SelectionChangeFcn',...
    @groupBtnTag_SelectionChangeFcn);
set(handles.colormapTag,'SelectionChangeFcn',...
    @colormapTag_SelectionChangeFcn);

% Update handles structure
set(handles.figure1,'Name',...
    'Computed tomography simulation by Nasser Abbasi, CSUF. EE 518, Digital Signal Processing');
guidata(hObject, handles);

process(hObject,0);

end

% --- Outputs from this function are returned to the command line.
function varargout = nma_projection_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
end


%%%%%%%%%%%%%%%%%%
%
%
%%%%%%%%%%%%%%%%%%%
function groupBtnTag_SelectionChangeFcn(hObject, eventdata)
process(hObject,0);
end

%%%%%%%%%%%%%%%%%%
%
%
%%%%%%%%%%%%%%%%%%%
function colormapTag_SelectionChangeFcn(hObject, eventdata)
handles   = guidata(hObject);
h         = get(handles.colormapTag);
hColorMap = h.SelectedObject;
switch get(hColorMap,'Tag')
    case 'jetTag'
        colormap(jet);
    case 'grayTag'
        colormap(gray);
end

end

%%%%%%%%%%%
%
%%%%%%%%%%%
function process(hObject,doSimulation)

%retrieve GUI data, i.e. the handles structure
handles = guidata(hObject);

nPixels  = 600; 
radius   = 120;
c        = 22
clip     = 10;

h         = get(handles.groupBtnTag);
hselected = h.SelectedObject;

%Now obtain colormap
h          = get(handles.colormapTag);
hColorMap  = h.SelectedObject;
switch get(hColorMap,'Tag')
    case 'jetTag'
        colormap(jet);
    case 'grayTag'
        colormap(gray);
end

%Now obtain the angle of projection and display projection
h     = get(handles.angleTag);
angle = str2double(h.String);

switch get(hselected,'Tag')   % Get Tag of selected object
    case 'blackDiskTag'
        A = makeDisk(0,nPixels,radius);
        p = radon(A,angle);

   case 'verticalBarTag'
        [A,p] = makeVerticalBar(nPixels);

    case 'randomDiskTag'
        A = makeDisk(1,nPixels,radius);
        p = radon(A,angle);
        
    case 'lenaTag'        
        fileName = 'lena.jpg';        
        A        = imread(fileName,'jpg');        
        p        = radon(A,angle);     
        
    case 'lungTag'        
        fileName = 'lung.jpg';        
        A        = imread(fileName,'jpg');        
        [nRow,nCol] = size(A);
        d = min(nRow,nCol);
        A = imresize(A, [d d]);
        p = radon(A,angle);  
        
     case 'blobsTag'        
        fileName     = 'blobs.gif';        
        A            = imread(fileName,'gif');        
        [nRow,nCol]  = size(A);
        d            = min(nRow,nCol);
        A            = imresize(A, [d d]);
        p            = radon(A,angle);
        
    otherwise
        % Code for when there is no match.
end

axes(handles.originalImageAxes);
imagesc(A);
axis(handles.originalImageAxes,'image');

%now display projection. 
axes(handles.projectionAxes);
stairs(p);

%now make fft of projection
axes(handles.normalProjectionTransformAxes);
YProjection = fft(p);
YshiftedProjection = fftshift(YProjection);
plot(abs(YshiftedProjection)); 

%now make 2D fft of original image
Y2D = fft2(double(A));
Y2Dshifted = abs(fftshift(Y2D));
axes(handles.twoDTransformAxes);
imagesc(c*log(1+Y2Dshifted));

xlabel('v'); ylabel('u');
axis(handles.twoDTransformAxes,'image');

%
%display the 2D spectrum on 3D
make3dspectrum(handles.FFT2on3DAxes,abs(Y2Dshifted),clip,c);

if doSimulation
    backProjection(hObject,A,clip,c);
end

end
%%%%%%%%%%%
%
%%%%%%%%%%%
function disk=makeDisk(isRandom,nPixels,r)
BLACK =0;
WHITE =255;

disk    = zeros(nPixels);
xCenter = ceil(nPixels/2);
yCenter = ceil(nPixels/2);

for i=1:nPixels
    for j = 1:nPixels
        xReal = i-xCenter;
        yReal = j-yCenter;
        distance = sqrt(xReal^2+yReal^2);
        if distance>r
           disk(i,j) = BLACK;
        else
           if isRandom
              disk(i,j) = WHITE*rand;
           else
              disk(i,j) = WHITE;
           end
        end
    end
end
end

%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%
function angleTag_Callback(hObject, eventdata, handles)
% hObject    handle to angleTag (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of angleTag as text
%        str2double(get(hObject,'String')) returns contents of angleTag as a double

process(hObject,0);
end

%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function angleTag_CreateFcn(hObject, eventdata, handles)
% hObject    handle to angleTag (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end

%%%%%%%%%%
%
%%%%%%%%%%
function [A,p]=makeVerticalBar(nPixels)
A=[0 0 1 0 0;
   0 0 1 0 0;
   0 0 1 0 0;
   0 0 1 0 0;   
   0 0 1 0 0];

p=[1;
   1;
   1;
   1;   
   1];
end

%%%%%%%%%%%%
%
%%%%%%%%%%%%%
function nAnglesTag_Callback(hObject, eventdata, handles)
% hObject    handle to nAnglesTag (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nAnglesTag as text
%        str2double(get(hObject,'String')) returns contents of nAnglesTag as a double

end

% --- Executes during object creation, after setting all properties.
function nAnglesTag_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nAnglesTag (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end

% --- Executes on button press in BtnTag.
function BtnTag_Callback(hObject, eventdata, handles)
% hObject    handle to BtnTag (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%Now obtain how many angles to use for backprojection
process(hObject,1);
end
%%%%%%%%%%
%
%%%%%%%%%%%
function backProjection(hObject,A,clip,c)

%retrieve GUI data, i.e. the handles structure
handles = guidata(hObject);

%Now obtain how many angles to use for backprojection
h       = get(handles.nAnglesTag);
nAngles = str2double(h.String);

angles = zeros(nAngles,1);
delta=180/(nAngles+1)
for i=1:nAngles
    angles(i)=delta*i;
end

R      = radon(A,angles);

[the_intepolation,the_filter] = get_iradon_options(handles);

for i = 1:nAngles
    I = iradon(R(:,1:i),angles(1:i),the_intepolation,the_filter);
    axes(handles.radonSimAxes);    
    imagesc(I);
   
    xlabel(sprintf('number of angles [%d]',i));
    
    make3dspectrum(handles.FFT2on3DcurrentAxes,abs(fftshift(fft2(I))),clip,c)        
end

end


% --- Executes on selection change in iradonInterpTag.
function iradonInterpTag_Callback(hObject, eventdata, handles)
% hObject    handle to iradonInterpTag (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns iradonInterpTag contents as cell array
%        contents{get(hObject,'Value')} returns selected item from iradonInterpTag
end

% --- Executes during object creation, after setting all properties.
function iradonInterpTag_CreateFcn(hObject, eventdata, handles)
% hObject    handle to iradonInterpTag (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end

% --- Executes on selection change in iradonFilterTag.
function iradonFilterTag_Callback(hObject, eventdata, handles)
% hObject    handle to iradonFilterTag (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns iradonFilterTag contents as cell array
%        contents{get(hObject,'Value')} returns selected item from iradonFilterTag
end

% --- Executes during object creation, after setting all properties.
function iradonFilterTag_CreateFcn(hObject, eventdata, handles)
% hObject    handle to iradonFilterTag (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end

