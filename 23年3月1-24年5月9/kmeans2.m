%% 1. 读取数据
data = xlsread('cgm_data.xlsx','Sheet2');   % N×T
[N, T] = size(data);
t = (0:T-1);  % 时间 index，真实分钟也可以

%% 2. 提取动力学特征（在原来的基础上补充斜率）

% 2.1 基线
baseline = mean(data(:,1:min(3,T)), 2);

% 2.2 峰值及达峰时间
[peak_val, peak_idx] = max(data, [], 2);

% 2.3 末尾
end_val = data(:,end);

% 2.4 上升幅度 / 速度
amp_up    = peak_val - baseline;
rise_time = max(peak_idx - 1, 1);
speed_up  = amp_up ./ rise_time;

% 2.5 下降幅度 / 速度
amp_down   = peak_val - end_val;
fall_time  = max(T - peak_idx, 1);
speed_down = amp_down ./ fall_time;

% 2.6 整体水平
mean_BG = mean(data, 2);

% 2.7 对整条曲线做标准化后做 PCA，提形状
data_norm = data - mean(data, 2);
data_std  = std(data_norm, 0, 2);
data_std(data_std == 0) = 1;
data_norm = data_norm ./ data_std;

numPC = min(3, T-1);
[coeff, score] = pca(data_norm, 'NumComponents', numPC);

pc1 = score(:,1);
pc2 = score(:,min(2,numPC));
pc3 = score(:,min(3,numPC));

% 2.8 早期 / 晚期斜率（用分段均值差近似）
w1 = 1:round(T/3);
w2 = round(T/3)+1:round(2*T/3);
w3 = round(2*T/3)+1:T;

m1 = mean(data(:,w1),2);
m2 = mean(data(:,w2),2);
m3 = mean(data(:,w3),2);

slope_early = (m2 - m1) / (mean(w2) - mean(w1));   % 前半段趋势
slope_late  = (m3 - m2) / (mean(w3) - mean(w2));   % 后半段趋势

%% 3. 组成特征矩阵：达峰时间 + 形状PC + 斜率 + 其他
% 列：baseline, peak_val, amp_up, speed_up, amp_down, speed_down,
%     mean_BG, peak_idx, pc1, pc2, pc3, slope_early, slope_late
features = [baseline, peak_val, amp_up, speed_up, ...
            amp_down, speed_down, mean_BG, ...
            peak_idx, pc1, pc2, pc3, slope_early, slope_late];

% 标准化
features_z = zscore(features);

% ====== 关键：加权，让"达峰时间 + 斜率"更重要 ======
w = ones(1, size(features_z,2));
% peak_idx 在第 8 列；早/晚斜率在第 12,13 列
w(8)  = 2.5;   % 达峰时间权重加大
w(12) = 2.0;   % 早期斜率
w(13) = 2.0;   % 晚期斜率

features_z = features_z .* w;   % 按列加权




%% 4. K-means 聚类（分 k 类）
k = 5;
[idx, C] = kmeans(features_z, k, 'Replicates', 50);

%% 4.1 计算每个样本到所属簇中心的距离
N = size(features_z,1);
dist_to_center = zeros(N,1);
for i = 1:N
    dist_to_center(i) = norm(features_z(i,:) - C(idx(i),:));  % 欧氏距离
end

%% 4.2 自动判定"离群点"并剔除
% 这里用全局均值 + 2*标准差作为阈值（可以自己调，比如 1.5 或 2.5）
mu    = mean(dist_to_center);
sigma = std(dist_to_center);
th    = mu + 1*sigma;        % 阈值

keep_mask   = dist_to_center <= th;   % 保留的样本
remove_mask = ~keep_mask;             % 被视为"脏"的样本

n_total  = N;
n_keep   = sum(keep_mask);
n_remove = sum(remove_mask);

fprintf('总样本数: %d，其中去掉离群点 %d (%.1f%%)，保留 %d。\n', ...
    n_total, n_remove, 100*n_remove/n_total, n_keep);

% 每一类各自删了多少
for c = 1:k
    n_c    = sum(idx == c);
    n_c_rm = sum(idx == c & remove_mask);
    fprintf('  Cluster %d: 原始 %d，删掉 %d (%.1f%%)\n', ...
        c, n_c, n_c_rm, 100*n_c_rm / max(n_c,1));
end

% 如果你想知道具体删掉哪些样本的行号：
removed_idx = find(remove_mask);   % 这些是原数据中被过滤掉的行

%% 4.3 对"干净数据"重新聚类（可选，但一般建议再跑一次）
features_clean = features_z(keep_mask,:);
data_clean     = data(keep_mask,:);

[idx_clean, C_clean] = kmeans(features_clean, k, 'Replicates', 50);


%% 5. 可视化：只画保留样本的聚类结果
figure;
for c = 1:k
    subplot(1,k,c);
    this_cluster = (idx_clean == c);
    plot(data_clean(this_cluster,:)', 'Color', [0.7 0.7 0.7]); hold on;
    plot(mean(data_clean(this_cluster,:),1), 'r','LineWidth',2);
    title(sprintf('Type %d  (n=%d)', c, sum(this_cluster)));
    xlabel('时间点'); ylabel('血糖值');
end
sgtitle(sprintf('过滤离群点后的聚类结果 (k=%d)', k));

%% 6. 基于"形状"的二次离群点过滤（在每个 cluster 内部做）

[M, T] = size(data_clean);
k = max(idx_clean);

% 先把每条曲线做形状标准化：减均值、除标准差
data_shape = data_clean - mean(data_clean, 2);
std_each   = std(data_shape, 0, 2);
std_each(std_each == 0) = 1;
data_shape = data_shape ./ std_each;

shape_keep_mask = true(M,1);   % 先默认都保留

% 每个 cluster 单独判断"形状离群"
remove_ratio = 0.10;   % 比如每类踢掉距离最远的10%，你可以改成 0.05 / 0.15 等

for c = 1:k
    idx_c = find(idx_clean == c);
    Xc = data_shape(idx_c,:);          % 该类的标准化曲线
    mean_shape = mean(Xc, 1);         % 该类的平均形状

    % 每条曲线到平均形状的欧氏距离
    d = sqrt(sum((Xc - mean_shape).^2, 2));

    % 找到本类最"远"的那一部分
    n_c = numel(idx_c);
    n_rm_c = round(remove_ratio * n_c);   % 本类要删掉的数量
    if n_rm_c > 0
        [~, order] = sort(d, 'descend');  % 从远到近
        rm_idx_local = order(1:n_rm_c);   % 这些是本类中要删的
        shape_keep_mask(idx_c(rm_idx_local)) = false;
    end
end

% 统计删除情况
M_final   = sum(shape_keep_mask);
M_removed = M - M_final;

fprintf('形状过滤阶段：在当前 %d 条中又删掉 %d 条 (%.1f%%)\n', ...
    M, M_removed, 100*M_removed/M);

for c = 1:k
    n_c    = sum(idx_clean == c);
    n_c_rm = sum(idx_clean == c & ~shape_keep_mask);
    fprintf('  Cluster %d: 原本 %d，形状过滤删掉 %d (%.1f%%)\n', ...
        c, n_c, n_c_rm, 100*n_c_rm/max(n_c,1));
end

% 得到最终保留的数据和标签
data_final = data_clean(shape_keep_mask,:);
idx_final  = idx_clean(shape_keep_mask);

%% 7. 再画一次最终结果
figure;
for c = 1:k
    subplot(1,k,c);
    this_cluster = (idx_final == c);
    plot(data_final(this_cluster,:)', 'Color', [0.7 0.7 0.7]); hold on;
    plot(mean(data_final(this_cluster,:),1), 'r', 'LineWidth', 2);
    title(sprintf('Type %d (n=%d)', c, sum(this_cluster)));
    xlabel('时间点'); ylabel('血糖值');
end
sgtitle(sprintf('特征+形状双重过滤后的聚类结果 (k=%d)', k));
