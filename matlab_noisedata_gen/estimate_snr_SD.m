function [snr_est,snr] = estimate_snr_SD(r)
% estimate_snr_SD - 基于SD算法的盲信噪比估计
%
% 输入：
%   r - 接收信号向量（复数或实数）
%
% 输出：
%   snr_est - 估计的SNR（线性值）
%
% 参考文献：
%   D. C. Rife and R. R. Boorstyn, "Single-tone parameter estimation from 
%   discrete-time observations," IEEE Trans. Info. Theory, 1974.
%
%   M. K. Simon and M.-S. Alouini, "Digital Communication over Fading Channels," 
%   Wiley, 2005.

%     r = r(:); % 确保为列向量
    r = r(1:1e6);  % 确保为列向量
    M2 = mean(abs(r).^2);
    M4 = mean(abs(r).^4);
    M6 = mean(abs(r).^6);

    numerator = M6 - 9*M2*M4 + 12*M2^3;
    denominator = 9*M4 - 15*M2^2;
    snr_est = numerator / denominator;

    snr = 10 * log10(snr_est);
end
