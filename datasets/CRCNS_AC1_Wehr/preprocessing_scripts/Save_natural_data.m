% ANALYSIS SAMPLE COMMANDS

param.homedir = '/mnt/Data1/audio_electrophysiology_datasets/CRCNS-AC1/crcns-ac1/wehr/'; % home directory
cd([param.homedir,'Tools']); % if needed

datafolders = dir(strcat(param.homedir, 'Results/2*'));
for folder = datafolders'
    
    % get experimment ID
    date = folder.name(1:8);
    penet = str2num(folder.name(13:15));
    
    % 1: tuning curve 
    ID.date = date; ID.PenetNo = penet;
    %TC = MakeTuningCurve(ID,param);
    
    % 2: response to natural sound sequence 
    param.fig = 1; % plot responses to the first stimulus
    data = SaveNaturalResponse(ID,param); % obtain stimulus and evoked responses
    
    % 3: select the preprocessed data
    % N is the number of sound clips
    % T is the number of samples
    % R is the number of repeats
    responses_natural = data.response.evoked;    % 1xN cell of TxR double
    spectros_natural = data.spectros;            % 1xN cell of TxF double
    
    savedirname = strcat(param.homedir, "Outputs/");
    savefilename = strcat(folder.name, "_natural.mat");
    save(strcat(savedirname, savefilename), "responses_natural", "spectros_natural");
    
end

disp("done")
