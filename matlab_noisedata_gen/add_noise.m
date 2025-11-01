function [rx_noisy] = add_noise(signal, snr_dB)
    rx_signal = signal(:);   % 原始信号
    % P_signal = mean(abs(rx_s).^2);      % 信号段功率
    % P_noise_orig = mean(abs(rx_n).^2);  % 原始噪声功率
    P_signal = 2.0661e-04;
    P_noise_orig = 2.2562e-05;
    
        
    % 计算需要额外加的噪声功率
    P_noise_add = P_signal / 10^(snr_dB/10) - P_noise_orig;
    
    % 如果计算结果小于0，说明原噪声已经比目标SNR大
    if P_noise_add < 0
        warning('目标SNR低于原始信号SNR，不能降低');
        rx_noisy = rx_signal; 
    else
        % 生成复数高斯噪声
        noise_add = sqrt(P_noise_add/2) * (randn(size(rx_signal)) + 1i*randn(size(rx_signal)));
        
        % 得到新的信号
        rx_noisy = rx_signal + noise_add;
        
        % 可选：验证实际SNR
        P_total_noise = P_noise_orig + P_noise_add;
        snr_check = 10*log10(P_signal / P_total_noise);
        fprintf('Target SNR = %2.1f dB, Achieved SNR = %.2f dB\n', snr_dB, snr_check);
    end
%     end
end