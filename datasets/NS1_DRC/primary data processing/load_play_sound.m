% Get the directory of sound files
soundFiles=dir('spikesandwav/SH.En.C');
% get rid of current and parent directory entries
soundFiles(1:2)=[];

% Read and play the sound for the Nth stimulus
N=12;
[y,fs]=audioread(['spikesandwav/SH.En.C/' soundFiles(N).name]);
sound(y,fs)