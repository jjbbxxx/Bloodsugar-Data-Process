%% 1. 读取数据
data = xlsread('cgm_data.xlsx','Sheet2');   % 452 × T
[N, T] = size(data);
fprintf('数据维度：%d 条序列 × %d 个时间点\n', N, T);

%% 2. 计算 DTW 距离矩阵（N × N）
distMatrix = zeros(N, N);

disp('正在计算 DTW 距离矩阵（可能需要一些时间）...');
for i = 1:N
    for j = i+1:N
        d = dtw(data(i,:), data(j,:));   % MATLAB 自带 dtw()
        distMatrix(i,j) = d;
        distMatrix(j,i) = d;
    end
end
disp('DTW 距离计算完成！');

%% 3. 层次聚类（Hierarchical clustering）
Z = linkage(distMatrix, 'average');  % 可改为 'complete' 或 'ward'

%% 4. 聚成 3 类
k = 3;
idx = cluster(Z, 'maxclust', k);

%% 5. 可视化：树状图
figure;
dendrogram(Z, 50);
title('层次聚类 Dendrogram（基于 DTW 距离）');

%% 6. 可视化：每类的平均曲线
figure;
for c = 1:k
    subplot(1,k,c);
    plot(data(idx==c,:)', 'Color', [0.7 0.7 0.7]); hold on;
    plot(mean(data(idx==c,:),1), 'r', 'LineWidth', 2);
    title(['Cluster ' num2str(c) ' （n=' num2str(sum(idx==c)) '）']);
    xlabel('时间点'); ylabel('血糖值');
end
sgtitle('三类时序聚类（DTW）');