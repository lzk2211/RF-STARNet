close all;

% strs = ["0000","0001","0010","0011","0100","0101","0110","0111"];
strs_mult = ["0010","0011","0101","0110","0111","1001"];
strs_58 = ["1010","1100","1110","1111","10010"];
strs_24 = ["10000","10001","10011","10100","10101","10110","10111","11000"];

strs_mult_12 = ["1011","1101"];

% 输入你想生成的类别名称
% strs = ["1010","1011","1100","1101","1110","1111"];
% strs = ["10000","10001","10010","10011","10100","10101","10110","10111","11000"];
% strs = ["0000","0001","0010","0011","0100","0101","0110","0111","1000","1001",...
%     "1010","1011","1100","1101","1110","1111","10000","10001","10010","10011","10100","10101","10110","10111","11000"];

strs = ["0010"];

% strs = "0010";
distance = "D00";
pro = "data_set_5e6_-5dB";
T = 5e6;
N = 2048;
fs = 1e8;
h = gpuArray(hanning(N)); % 将汉宁窗转移到GPU
noverlap = N/2;

% 检查GPU是否可用
if gpuDeviceCount == 0
    error('No GPU device found. Please check your GPU configuration.');
else
    gpu = gpuDevice();
    fprintf('Using GPU: %s\n', gpu.Name);
end

for k = 1:length(strs)
    str = strs(k);
    filename = sprintf('I:\\%s\\mat\\T%s\\*.mat', distance, str);
    if exist(sprintf('I:\\%s\\%s\\T%s', distance, pro, str))==0 %%判断文件夹是否存在
        mkdir(sprintf('I:\\%s\\%s\\T%s', distance, pro, str));  %%不存在时候，创建文件夹
    end
    
    Files = dir(fullfile(filename));
    LengthFiles = length(Files);
    
    index = 0;
    
    for j=1:LengthFiles
        name=Files(j).name;           %读取struct变量的格式
        folder=Files(j).folder;
        disp(['loading... ' folder,'\\',name]);
        load([folder,'\\',name]);    %导入文件

%% 筛选不同的信号取不同的通道
        if any(strcmp(str, strs_mult))
            if name(end-7)=='0'%前8
                signal = RF0_I + 1i*RF0_Q;
            else%后8
                signal = RF1_I + 1i*RF1_Q;
            end
        elseif any(strcmp(str, strs_58))
            signal = RF1_I + 1i*RF1_Q;
        elseif any(strcmp(str, strs_24))
            signal = RF0_I + 1i*RF0_Q;
        elseif any(strcmp(str, strs_mult_12))
            if str == "1101"
                if bin2dec(name(end-7:end-4)) <= 5
                    signal = RF1_I + 1i*RF1_Q;
                else
                    signal = RF0_I + 1i*RF0_Q;
                end
            end
            if str == "1011"
                if bin2dec(name(end-7:end-4)) > 6
                    signal = RF1_I + 1i*RF1_Q;
                else
                    signal = RF0_I + 1i*RF0_Q;
                end
            end
        else
            signal = RF0_I + 1i*RF0_Q;
        end

        signal = add_noise(signal, -5);
%%
        % 将信号数据转移到GPU
        signal = gpuArray(signal);
        
        disp('packing pic...');
        for i = 0:(size(signal)/T)-1
            signal_data = signal((i*T+1):((i+1)*T));
            
            % 在GPU上执行STFT计算
            [S, f, t] = spectrogram(signal_data, h, noverlap, N, fs);
            S = fftshift(S);
            
            % 将结果转移回CPU
%             A = abs(S);
            A = 20*log10(abs(S));
        
%             mat_name = sprintf('H:\\%s\\%s\\T%s\\%d.mat', distance, pro, str, index + i);
            pic_name = sprintf('I:\\%s\\%s\\T%s\\%d.png', distance, pro, str, index + i);
            
            
            % 归一化矩阵
            A_min = min(A(:)); % 矩阵的最小值
            A_max = max(A(:)); % 矩阵的最大值
            
%             计算最大值和最小值之间的差
            A_range = A_max - A_min;
            
%             归一化
            A_normalized = (A - A_min) / A_range;
    
            GRAY_image = cat(3, A_normalized, A_normalized, A_normalized);
    
            % 保存图像
%             save(mat_name, "A");
            
            GRAY_image = imresize(GRAY_image, [512,512]);
            GRAY_image = single(GRAY_image);
            GRAY_image = gather(GRAY_image);
            imwrite(GRAY_image, pic_name);% 保存的时候需要归一化
%             writeNPY(A, mat_name);
%             save(mat_name, 'A');
%             save(pic_name, 'A_normalized')
            fprintf('%d ', index + i);
        end
        fprintf('\n');
        index = index + i + 1;    
    end

end