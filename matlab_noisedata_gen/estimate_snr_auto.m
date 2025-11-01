function snr_db = estimate_snr_auto(rx, varargin)
%==============================================================
% 功能：自动估计信号的信噪比（SNR）
%
% 输入：
%   rx       - 输入信号（复数或实数向量）
%   'noise_segment' - 可选，纯噪声段
%   'win'           - 可选，短时能量窗口长度（默认1024）
%   'quantile'      - 可选，判定静默窗口的比例（默认0.05）
%
% 输出：
%   snr_db   - 估计得到的信噪比（单位 dB）
%
% 使用示例：
%   snr_db = estimate_snr_auto(rx);                  % 自动估计
%   snr_db = estimate_snr_auto(rx, 'noise_segment', noise_only);
%==============================================================

p = inputParser;
addParameter(p,'noise_segment',[]);
addParameter(p,'win',1024);
addParameter(p,'quantile',0.05);
parse(p,varargin{:});

rx = rx(:); % 转列向量
win = p.Results.win;
q = p.Results.quantile;
noise_segment = p.Results.noise_segment;

%% 1️⃣ 如果提供了噪声段，直接计算噪声功率
if ~isempty(noise_segment)
    noise_segment = noise_segment(:);
    P_noise = mean(abs(noise_segment).^2);
else
    %% 2️⃣ 没有噪声段，则自动选择短时能量最低的窗口估算噪声
    N = length(rx);
    numBlocks = floor(N/win);
    E = zeros(numBlocks,1);
    for k = 1:numBlocks
        blk = rx((k-1)*win+1 : k*win);
        E(k) = mean(abs(blk).^2);
    end
    th = quantile(E, q);
    noiseBlocks = E <= th;
    P_noise = mean(E(noiseBlocks));
end

%% 3️⃣ 计算总功率
P_total = mean(abs(rx).^2);

%% 4️⃣ 信号有效功率
P_signal = max(P_total - P_noise, eps);

%% 5️⃣ 计算SNR
snr_db = 10*log10(P_signal / P_noise);

fprintf('Estimated SNR = %.2f dB\n', snr_db);

end
