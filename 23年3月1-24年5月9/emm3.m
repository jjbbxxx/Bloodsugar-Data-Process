data = xlsread('cgm_data.xlsx','Sheet2');   % N×T
[N, T] = size(data);

% 三段时间窗口
w1 = 1:round(T/3);
w2 = round(T/3)+1:round(2*T/3);
w3 = round(2*T/3)+1:T;

m1 = mean(data(:, w1), 2);   % 早期均值
m2 = mean(data(:, w2), 2);   % 中期均值
m3 = mean(data(:, w3), 2);   % 晚期均值

s12 = m2 - m1;   % 前半段"斜率"：中 - 早
s23 = m3 - m2;   % 后半段"斜率"：晚 - 中

amp = max(data,[],2) - min(data,[],2);   % 整体振幅

% 振幅阈值：大于这个才算"明显变化"
amp_th = median(amp);   

type = zeros(N,1);  % 1=峰型, 2=下降型, 3=其它

for i = 1:N
    if amp(i) > amp_th && s12(i) > 0 && s23(i) < 0
        % 先升后降 & 振幅大 → 峰型
        type(i) = 1;
    elseif amp(i) > amp_th && s12(i) < 0 && s23(i) < 0
        % 一直往下 & 振幅大 → 高基线下降型
        type(i) = 2;
    else
        % 其他：平平台、小波动、乱动 → 杂型
        type(i) = 3;
    end
end

figure;
k = 3;
for c = 1:k
    subplot(1,k,c);
    idx = (type == c);
    Xc = data(idx,:);
    plot(Xc','Color',[0.7 0.7 0.7]); hold on;
    plot(mean(Xc,1),'r','LineWidth',2);
    title(sprintf('Type %d (n=%d)', c, sum(idx)));
    xlabel('时间点'); ylabel('血糖值');
end
sgtitle('按斜率规则分型（先升后降/整体下降/其它）');
