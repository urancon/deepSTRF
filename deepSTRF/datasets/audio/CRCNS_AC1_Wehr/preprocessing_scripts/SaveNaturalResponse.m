function p = ViewResponse(ID,PARAM)

% displays responses aligned with stimuli
%
% --- Input ---
%   ID - structure containing 
%       .date : experiment date (yyyymmdd)
%       .PenetNo : Penetration Number
%
%   param - structure containing various parameters
%       .smooth : window length of median filter smoothing (default: 10 msec)
%       .fig : vector of stimulus index to be shown
%       .homedir : home directory
%
% --- Output ---
%   p - structure containing 
%       .evoked : cell containing evoked-responses over trials
%       .spont  : cell containing spontaneous activity over trials
%       .stimulus : structure containing corresponding stimuli
%           .description : natural sound description
%           .trial : number of trials tested
%           .samples : signal
%           .sf : sampling frequency
%           .ID : structure of experiment information
%   
% See also GetDataInfo.m, GetEvoked.m, MakeRamp.m
%

%% version information
%   Hiroki Asari, Zador Lab, CSHL.
%   Revision.1 (2007/12/06): open to the lab


%% obtain experimental data information %%%%%
error(nargchk(1,2,nargin));
if nargin<2, PARAM = struct;end
P = GetDataInfo(ID,PARAM);


%% index for 'natural sound' stimuli %%%%%
I = false(size(P.stimulus)); trial = zeros(size(P.stimulus));
for i=1:length(P.stimulus),
    if ~strcmpi(P.stimulus(i).description,'tuning curve') && ...
       ~strcmpi(P.stimulus(i).description,'silence') && ...
       ~strcmpi(P.stimulus(i).description,'white noise'),
           I(i) = true;
    end
    trial(i) = P.stimulus(i).trial;
end
trial = [cumsum([1,trial(1:end-1)]);cumsum(trial)];
trial = trial(:,I); stim = P.stimulus(I);
PARAM.homedir = P.homedir;
clear I i P


%% load responses over trials %%%%%
if isfield(PARAM,'smooth') && ~isempty(PARAM.smooth), smooth = PARAM.smooth;
else                                                  smooth = 10; % msec window
end
if isfield(PARAM,'offset') && ~isempty(PARAM.offset), offset = PARAM.offset;
else                                                  offset = 0; % no offset responses
end
Evoked = cell(size(stim)); Spont = Evoked;
for i=1:length(stim), 
    for j=trial(1,i):trial(2,i),
        filename = [PARAM.homedir,'Results',filesep,ID.date,'-mw-00',...
            num2str(ID.PenetNo),filesep,ID.date,'-mw-00',num2str(ID.PenetNo),...
            '-',num2str([zeros(1,floor(3-log10(j+1))),j]),'.mat'];
        filename = strrep(strrep(filename(~isspace(filename)),'\',filesep),'/',filesep);
        load(filename);
    
        %% divide responses into evoked and spontaneous period
        param.smooth = smooth; param.offset = offset;
        [evoked,spont,param] = GetEvoked(response,triggers,param);
        Evoked{i} = [Evoked{i},evoked{1}]; Spont{i} = [Spont{i},spont{1}];
    end
    Evoked{i} = Evoked{i}'; Spont{i} = Spont{i}';
    
    %% load stimulus
    % commented original line because of an error and fixed it below
    %load(strrep(strrep([PARAM.homedir,'Stimuli',filesep,triggers.file]),'/',filesep),'\',filesep);
    load(strrep(strrep([PARAM.homedir,'Stimuli',filesep,triggers.file],'/',filesep),'\',filesep));
    stim(i).samples = MakeRamp(stimulus.samples,triggers.param);
    stim(i).sf = stimulus.param.sf;
end
sf = response.sf;
clear i j spont evoked param response triggers stimulus


%% output %%%%%
if nargout>0,
    p.response.evoked = Evoked;
    p.response.spont = Spont;
    p.response.sf = sf;
    for i=1:length(stim),
        stim(i).ID = ID;
        stim(i).ID.RecNo = trial(1,i):trial(2,i);
    end
    p.stimulus = stim;
end


%% save (no figure) %%%%%
spectrograms = {};

if isfield(PARAM,'fig') && ~isempty(PARAM.fig),
    cmap = [-50 50]; % colormap
    dv = 20; % mV
    %for i = unique(ceil(PARAM.fig(PARAM.fig>0 & PARAM.fig<=length(Evoked)))),
    for i = 1:length(Evoked)
        [n,m] = size(Evoked{i});
        
        [S,t,x,f] = TFdomain(stim(i).samples,stim(i).sf,PARAM);
        
        spectrograms = [spectrograms, S];  % save spectrogram
        
        
%         figure;subplot(2,1,1); % stimulus
%         imagesc(t+t(2)/2,x,S,cmap);axis xy;colormap(jet);        
%         j = find(x==round(x/2)*2,2); % every 2 octaves 
%         set(gca,'YTick',x(j(1):diff(j):end),'YTickLabel',f(j(1):diff(j):end)/1000,...
%             'TickDir','out');
%         ylabel('Frequency [kHz]'); 
%         %xlabel('Time [sec]');
%         xlim([0 min(m/sf,t(end)+t(2))]);
        
%         subplot(2,1,2);hold on; % response
%         for j=1:n,  % for each repeat (n total)
%             plot([1:m]/sf,Evoked{i}(j,:)+(j-1)*dv,'b','LineWidth',2); % subthreshold
%         end
%         axis([0 min(m/sf,t(end)+t(2)) floor(min(Evoked{i}(1,:))/10)*10 ceil(max(Evoked{i}(end,:)+(n-1)*dv)/10)*10]);
%         
%         set(gca,'YTick',floor(min(Evoked{i}(1,:))/10)*10 + [0:dv:dv],...
%             'YTickLabel',floor(min(Evoked{i}(1,:))/10)*10 + [0:dv:dv],'TickDir','out');
%         ylabel('Response [mV]'); 
%         xlabel('Time [sec]');
    end
end


% save for output
p.spectros = spectrograms;
   
   

%%%%%% make spectrograms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [S,t,x,f] = TFdomain(stim,sf,param)
if isfield(param,'dt') && ~isempty(param.dt),   dt = param.dt;
else                                            dt = 0.001; % 0.02; % 20 msec
end
if isfield(param,'dx') && ~isempty(param.dx),   dx = param.dx;
else                                            dx = 1/6; % 6 bins/octave
end
if isfield(param,'overlap') && ~isempty(param.overlap), overlap = param.overlap;
else                                                    overlap = 2; % # of overlap window
end
p = zeros(round(dt*sf),1);
fmin = 100; fmax = 25600; % 100 Hz to 25.6 kHz spectrogram
[S,p,param] = logspectrogram([p;stim(:);p], sf, dt, dx, overlap, 0, fmin, fmax);
x = param.x; f = param.f; t = param.t;