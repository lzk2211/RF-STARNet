% -------- 信号段 --------
% load('H:\D00\mat\T0010\T0010_D00_S0000.mat');
% signal = RF0_I + 1i * RF0_Q;
% rx_s = signal(2498020:3408940);
load('H:\D00\mat\T1100\T1100_S0000.mat');
signal = RF1_I + 1i * RF1_Q;
rx_s = signal(1942800:2428240);
rx_s = rx_s(:);
P_signal = mean(abs(rx_s).^2);

% -------- 噪声段 --------
load('H:\D00\mat\T0000\T0000_D00_S0000.mat');

signal = RF1_I + 1i * RF1_Q;
rx_n = signal(1:1e5);
rx_n = rx_n(:);
P_noise = mean(abs(rx_n).^2);

% -------- 信噪比计算 --------
P_noise = max(P_noise, eps); % 防止分母为零
SNR = 10 * log10(P_signal / P_noise);

fprintf('Signal Power = %.2e\n', P_signal);
fprintf('Noise Power  = %.2e\n', P_noise);
fprintf('Estimated SNR = %.2f dB\n', SNR);
