clc; clear;

% Load NinaPro dataset file
load('S1_E1_A1.mat');

emg_data = emg;
labels = restimulus;

% Remove rest class (0)
valid_idx = labels ~= 0;
emg_data = emg_data(valid_idx,:);
labels = labels(valid_idx);

window_size = 200;
step_size = 200;

segments = [];
segment_labels = [];

for i = 1:step_size:(length(emg_data)-window_size)

    window = emg_data(i:i+window_size-1,:);
    label_window = labels(i:i+window_size-1);

    % Majority label in window
    label = mode(label_window);

    segments = cat(3, segments, window');
    segment_labels = [segment_labels; label];

end

save('emg_segments.mat','segments','segment_labels');

disp("Saved emg_segments.mat");
