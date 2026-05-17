%%
% Curated and adapted version from Patrick Mineault's github repository:
% https://github.com/patrickmineault/your-head-is-there-to-move-you-around
%
% Associated to the paper "Your head is there to move you around". 
% Preprint here: https://www.biorxiv.org/content/10.1101/2021.07.09.451701v2.abstract
% 
% Useful to genextract optic flow components necessary to generate kinematogram stimuli
% (cf. generate_crcns_hyperflow.py)
%
% Please adapt paths to your own system.
%
%%


% This reuses some of the scripts included in the crcns-mt1 dataset.
addpath('/mnt/e/data_derived/crcns-mt1/crcns-mt1-MatLab-scripts');
%%
basepath = '/mnt/e/data_derived/crcns-mt1/crcns-mt1-data/';
ds = dir([basepath '*.mat']);


%%
ii = 10;
cellname = ds(ii).name;
load([basepath cellname]);


%%
for ii = 1:length(ds)
    cellname = ds(ii).name;
    load([basepath cellname]);
    
    data = struct();
    data.name = cellname(1:end-4);
    data.stmatcheshf = false;
    
    width = round(3 * mean(std(mtdata.aperturecenter)) + mtdata.aperturediameter);

    disp(' Reconstructing the velocity field ...')
    params = struct('designsizex', width, 'designsizey', width, 'spatres', width/56,...
        'maskdiameter',mtdata.aperturediameter);
    Nx = round(params.designsizex/params.spatres(1));

    % Reconstruct the stimulus
    stimorg = GetVelField(params, mtdata.opticflows, mtdata.aperturecenter);

    % Downsample temporally 2-fold.
    eyeloc = (mtdata.eyeloc(1:2:end-1, :) + mtdata.eyeloc(2:2:end, :))/2;
    spkbinned = (mtdata.spkbinned(1:2:end-1) + mtdata.spkbinned(2:2:end));
    stimorg = (stimorg(1:2:end-1, :) + stimorg(2:2:end, :)) / 2;
    
    % Make sure most of the stimulus is within view.
    npixels = max(sum(abs(stimorg)>0, 2));
    validx = sum(abs(stimorg)>0, 2) > npixels / 2;
    
    validx = validx & (abs(eyeloc(:,1)) < 3) & (abs(eyeloc(:,2)) < 3);
    
    %data.
    % add a handful of times after for padding
    ntau = 3;
    Xidx = bsxfun(@plus, (1:size(stimorg, 1))' - 10 + ntau, (0:9));
    
    % Sampling grid
    rg = 0:params.spatres:params.designsizex;
    rg = rg(1:end-1);
    rg = rg - mean(rg);
    [xi, yi] = meshgrid(rg, rg);
    
    goodidx = validx & all((Xidx >= 1) & (Xidx <= size(stimorg, 1)), 2);
    
    fprintf('Cell %s, Good idx: %.3f, total samples %d\n', cellname, mean(goodidx), sum(goodidx));
    
    framerate = 500 / mtdata.dt;
    t = (1:size(stimorg,1))' / framerate;
    
    data.Y_hf = spkbinned(goodidx);
    data.stim_hf = stimorg;
    data.stimidx_hf = Xidx(goodidx, :);
    data.gridx_hf = xi;
    data.gridy_hf = yi;
    data.t = t(goodidx);
    
    save(sprintf('/mnt/e/data_derived/crcns-mt1/designmats/cell%02d.mat', str2double(cellname(5:end-4))), '-struct', 'data', '-v7.3');
end

