% -------- 信号段 --------
load('H:\D00\mat\T0010\T0010_D00_S0000.mat');
signal = RF0_I + 1i * RF0_Q;
% rx_s = signal(1270380:2523040);
% load('H:\D00\mat\T1100\T1100_S0000.mat');
% signal = RF1_I + 1i * RF1_Q;
% rx_s = signal(1942800:2428240);
% rx_s = rx_s(:);
% P_signal = mean(abs(rx_s).^2);
% 
% % -------- 噪声段 --------
% load('H:\D00\mat\T0000\T0000_D00_S0000.mat');
% 
% signal = RF1_I + 1i * RF1_Q;
% rx_n = signal(1:1e5);
% rx_n = rx_n(:);
% P_noise = mean(abs(rx_n).^2);

% fprintf('Signal Power = %.2e\n', P_signal);
% fprintf('Noise Power  = %.2e\n', P_noise);
% fprintf('Estimated SNR = %.2f dB\n', SNR);



threshold = 0.00689697;
signal = signal(1:1e5);
signal = signal(:);

% 计算信号幅度
mag = abs(signal);

% 找出大于阈值的点（信号部分）
signal_idx = mag > threshold;

% 找出小于等于阈值的点（噪声部分）
noise_idx = ~signal_idx;

% 计算信号功率
if any(signal_idx)
    signal_power = mean(abs(signal(signal_idx)).^2);
else
    warning('没有点超过阈值，信号功率为0');
    signal_power = 0;
end

% 计算噪声功率
if any(noise_idx)
    noise_power = mean(abs(signal(noise_idx)).^2);
else
    warning('所有点都超过阈值，噪声功率为0');
    noise_power = eps; % 防止除以零
end

% 计算信噪比
snr_linear = signal_power / noise_power;
snr_dB = 10 * log10(snr_linear);

% 打印结果
fprintf('信号功率: %.4e\n', signal_power);
fprintf('噪声功率: %.4e\n', noise_power);
fprintf('信噪比: %.2f dB\n', snr_dB);