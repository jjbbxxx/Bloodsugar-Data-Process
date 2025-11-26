import glob
import os

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

###############################
# 1. 基础配置（路径 & 病案号）
###############################

# 原始“检验”总表路径（请改成你自己的）
ORIGINAL_EXCEL_PATH = r"（1）原数据/230301-240509.xlsx"
RAW_CGM_DIR = r"（1）原数据/1_动态血糖"

# 动态血糖监测其他数据（额外的馒头餐来源）
OTHER_EXCEL_PATH = r"（1）原数据/1_动态血糖监测其他数据.xlsx"

# CGM 文件夹路径（处理后的 2465 人动态血糖）
CGM_DIR = r"（2）处理之后的数据/2465人动态血糖/"

# 要读取的文件个数
N_FILES = 20


###############################
# 2. 找出该病案号的馒头餐事件日期
###############################

def find_mantou_meals(original_excel_path, case_id):
    """
    在两个文件中查找馒头餐事件：
    1）230301-240509.xlsx 的“检验”sheet：
       - C 列：病案号
       - G 列：报告时间（含日期）
       - S、AB、AF 三列同时非空 → 一次馒头餐
    2）1_动态血糖监测其他数据.xlsx（默认用第 1 个 sheet）：
       - B 列：病案号
       - C 列：时间（含日期）
       - F、AC、AG 三列同时非空 → 一次馒头餐

    返回：该病案号所有馒头餐“就餐时间”（当日早 7:00）及对应静脉血葡萄糖值的列表
    """

    meal_dict = {}

    # ========= 1. 先查 230301-240509.xlsx 里的“检验” =========
    df = pd.read_excel(original_excel_path, sheet_name="检验")

    # 注意：iloc 是 0 开始，所以 C=2, G=6, S=18, AB=27, AF=31
    col_case = df.iloc[:, 2]   # C 列：病案号
    col_time = df.iloc[:, 6]   # G 列：报告时间
    col_S    = df.iloc[:, 18]  # S 列 (Glu)
    col_AB   = df.iloc[:, 27]  # AB 列
    col_AF   = df.iloc[:, 31]  # AF 列

    mask_case = (col_case == case_id)
    mask_lab  = col_S.notna() & col_AB.notna() & col_AF.notna()
    sub = df[mask_case & mask_lab].copy()

    if not sub.empty:
        report_times = pd.to_datetime(sub.iloc[:, 6])
        glu_values = sub.iloc[:, 18]
        for rt, glu in zip(report_times, glu_values):
            meal_dt = datetime.combine(rt.date(), time(hour=7, minute=0))  # 当日 7:00
            if meal_dt not in meal_dict:
                meal_dict[meal_dt] = float(glu)

    # ========= 2. 再查 1_动态血糖监测其他数据.xlsx =========
    try:
        df2 = pd.read_excel(OTHER_EXCEL_PATH, sheet_name="检验")  # 默认第一个 sheet
    except FileNotFoundError:
        print("未找到 1_动态血糖监测其他数据.xlsx，先只用检验表。")
        df2 = None

    if df2 is not None:
        # B 列：病案号 → iloc[:,1]
        # C 列：时间   → iloc[:,2]
        # F 列：检测值 → iloc[:,5]
        # AC 列：第 28 号列（A=0,B=1,...,Z=25,AA=26,AB=27,AC=28）
        # AG 列：第 32 号列（AG=32）
        col_case2 = df2.iloc[:, 1]   # B 列：病案号
        col_time2 = df2.iloc[:, 2]   # C 列：时间
        col_F  = df2.iloc[:, 5]      # F 列 (Glu)
        col_AC = df2.iloc[:, 28]     # AC 列
        col_AG = df2.iloc[:, 32]     # AG 列

        mask_case2 = (col_case2 == case_id)
        mask_lab2  = col_F.notna() & col_AC.notna() & col_AG.notna()
        sub2 = df2[mask_case2 & mask_lab2].copy()

        if not sub2.empty:
            report_times2 = pd.to_datetime(sub2.iloc[:, 2])
            glu_values2 = sub2.iloc[:, 5]
            for rt, glu2 in zip(report_times2, glu_values2):
                meal_dt = datetime.combine(rt.date(), time(hour=7, minute=0))
                if meal_dt not in meal_dict:
                    meal_dict[meal_dt] = float(glu2)

    # ========= 3. 去重 & 排序 =========
    if not meal_dict:
        print(f"没有在两个文件中找到病案号 {case_id} 的馒头餐记录")
        return []

    meal_infos = []
    for dt in sorted(meal_dict.keys()):
        meal_infos.append({"meal_datetime": dt, "venous_glu": meal_dict[dt]})

    return meal_infos

###############################
# 3. 计算 AUC / pAUC / nAUC / iAUC
###############################

def trapezoid_auc(t_minutes, g_values):
    """
    标准梯形法计算 AUC。
    t_minutes: array-like, 时间点（单位：min），如 [0,15,...,180]
    g_values:  array-like, 对应的血糖值
    返回：AUC
    """
    t = np.array(t_minutes, dtype=float)
    g = np.array(g_values, dtype=float)

    auc = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        auc += (g[i] + g[i-1]) / 2 * dt
    return auc


def compute_p_n_i_auc(G0, t_minutes, g_values):
    """
    根据 G0、时间点和血糖序列，计算 pAUC / nAUC / iAUC。
    定义：
    - 找到 0-180min 内曲线第一次回到 G0 的时间点 t*
    - pAUC: 0 → t* 的曲线下面积
    - nAUC: t* → 180 的曲线下面积
    - iAUC = pAUC - nAUC
    若整段都未回到 G0，则：
    - pAUC = 全段 AUC
    - nAUC = 0
    """

    t = np.array(t_minutes, dtype=float)
    g = np.array(g_values, dtype=float)

    # 先算总 AUC（0–180）
    total_auc = trapezoid_auc(t, g)

    # 从 t>0 开始查找“穿过 G0”的位置
    cross_index = None
    for i in range(1, len(t)):
        v1 = g[i-1] - G0
        v2 = g[i]   - G0
        # 若两端符号不同或有一端刚好等于 0，认为在该区间内发生了交叉
        if v1 == 0:
            cross_index = i-1
            t_cross = t[i-1]
            g_cross = G0
            break
        if v1 * v2 < 0 or v2 == 0:
            # 在线性段 [t[i-1], t[i]] 内插求解 g(t*) = G0
            t1, t2 = t[i-1], t[i]
            g1, g2 = g[i-1], g[i]
            # g1 + (g2-g1)*(x-t1)/(t2-t1) = G0
            if g2 != g1:
                ratio = (G0 - g1) / (g2 - g1)
                t_cross = t1 + ratio * (t2 - t1)
            else:
                t_cross = t1  # 极少数 g1=g2=G0
            g_cross = G0
            cross_index = i
            break

    if cross_index is None:
        # 整段未回到 G0：认为 pAUC=总面积, nAUC=0
        p_auc = total_auc
        n_auc = 0.0
        i_auc = p_auc - n_auc
        return p_auc, n_auc, i_auc

    # 现在已有 t_cross, g_cross=G0
    # 1) 先算 0 → t_cross 的面积（需要处理最后一小段 [t[cross_index-1], t_cross]）
    t_p = list(t[:cross_index]) + [t_cross]
    g_p = list(g[:cross_index]) + [g_cross]
    p_auc = trapezoid_auc(t_p, g_p)

    # 2) 再算 t_cross → 180 的面积
    t_n = [t_cross] + list(t[cross_index:])
    g_n = [g_cross] + list(g[cross_index:])
    n_auc = trapezoid_auc(t_n, g_n)

    i_auc = p_auc - n_auc
    return p_auc, n_auc, i_auc


###############################
# 4. 处理单次馒头餐事件
###############################

def process_one_meal(cgm_file_path, case_id, meal_datetime, venous_glu=None):
    """
    对某个病案号的一次馒头餐事件（就餐时间 meal_datetime）：
    - 读取该病人的 CGM 文件
    - 计算 G0（6:30-7:00 平均）
    - 计算 G15~G180（线性插值）
    - 找 G_Peak、达峰时间
    - 算两段斜率
    - 算 AUC / pAUC / nAUC / iAUC
    - 另外：提取餐前30分钟到餐后210分钟的原始CGM数据
    返回：
      result_dict: 各种指标
      raw_window:  该次馒头餐 [-30min, +210min] 的原始CGM子表
    """

    # 读取 CGM 文件（先尝试处理后的单人文件，若该时间窗口无数据，则回退到原始动态血糖文件）
    df = pd.read_excel(cgm_file_path, sheet_name=0)
    # 这里按 123134.xlsx 的格式：
    # 列：['住院号', '测量时间', '测量值']
    # 如果列名中包含“测量时间”“测量值”，直接使用；否则再按位置兜底（主要给后续扩展用）
    if "测量时间" in df.columns and "测量值" in df.columns:
        df["测量时间"] = pd.to_datetime(df["测量时间"])
    else:
        # 假如格式变化，这里可以按第2、3列作为时间和值（仅兜底，不建议依赖）
        df_time = pd.to_datetime(df.iloc[:, 1])
        df_value = df.iloc[:, 2]
        df = df.copy()
        df["测量时间"] = df_time
        df["测量值"] = df_value

    # 如果一个文件里有多个人，这里再按住院号筛一遍
    if "住院号" in df.columns:
        df = df[df["住院号"] == case_id].copy()
    df = df.sort_values("测量时间")

    # ------------ 时间窗口定义 ------------
    start_time = meal_datetime - timedelta(minutes=30)  # 6:30
    end_time_180 = meal_datetime + timedelta(minutes=180)  # 10:00  （这个继续给 G0-G180、AUC 用）

    # 用于插值和 AUC 的子集（6:30~10:00）——跟之前一样
    mask_window_180 = (df["测量时间"] >= start_time) & (df["测量时间"] <= end_time_180)
    sub = df[mask_window_180].copy()

    if sub.empty:
        pattern = os.path.join(RAW_CGM_DIR, f"{case_id}-*.xlsx")
        files = glob.glob(pattern)
        if not files:
            print(
                f"{case_id} 在 {meal_datetime.date()} 附近 6:30–10:00 在处理后CGM中无数据，且未在原始动态血糖文件中找到匹配文件：{pattern}")
            return None, None
        raw_path = files[0]
        df_raw = pd.read_excel(raw_path, sheet_name=0)
        # 原始动态血糖文件：时间在C列，测量值在D列
        time_col = pd.to_datetime(df_raw.iloc[:, 2])
        value_col = df_raw.iloc[:, 3]
        df = pd.DataFrame({"测量时间": time_col, "测量值": value_col}).sort_values("测量时间")

        # 重新基于原始CGM数据计算 6:30~10:00 窗口
        mask_window_180 = (df["测量时间"] >= start_time) & (df["测量时间"] <= end_time_180)
        sub = df[mask_window_180].copy()

        if sub.empty:
            print(f"{case_id} 在 {meal_datetime.date()} 附近 6:30–10:00 在处理后CGM和原始CGM中都没有数据")
            return None, None

    # 此时 df 已经确定为本次计算所用的CGM数据（处理后或原始），在这里做质控
    qc_status, qc_detail = qc_cgm_for_meal(df, meal_datetime, venous_glu=venous_glu)


    # ⭐ 原始数据窗口：从 6:30 开始往后 23 个数据
    cand = df[df["测量时间"] >= start_time].sort_values("测量时间")

    N_POINTS = 23  # 想改别的个数就改这里

    if len(cand) < N_POINTS:
        raw_window = cand.copy()
    else:
        raw_window = cand.iloc[:N_POINTS].copy()

    # -------- 1) 计算 G0：6:30–7:00 平均 --------
    mask_baseline = (sub["测量时间"] >= start_time) & (sub["测量时间"] <= meal_datetime)
    baseline = sub[mask_baseline]
    if baseline.empty:
        G0 = np.nan
    else:
        G0 = baseline["测量值"].mean()

    # -------- 2) 计算 G15~G180（线性插值，仍然用 6:30~10:00 的 sub） --------
    times = sub["测量时间"]
    values = sub["测量值"]

    # 相对就餐时间（min）
    t_rel = (times - meal_datetime).dt.total_seconds() / 60.0

    order = np.argsort(t_rel.values)
    t_rel = t_rel.values[order]
    values = values.values[order]

    # 0~180，每15min
    time_points = list(range(0, 181, 15))  # [0, 15, ..., 180]

    def interp_at(t_target):
        if t_target <= t_rel[0]:
            return values[0]
        if t_target >= t_rel[-1]:
            return values[-1]
        idx_after = np.searchsorted(t_rel, t_target)
        idx_before = idx_after - 1
        t1, t2 = t_rel[idx_before], t_rel[idx_after]
        g1, g2 = values[idx_before], values[idx_after]
        return g1 + (g2 - g1) * (t_target - t1) / (t2 - t1)

    G_points = {"G0": G0}
    for t in time_points[1:]:  # 跳过0，G0单独算
        G_points[f"G{t}"] = float(interp_at(t))

    # -------- 3) 峰值 & 达峰时间（0~180min，用原始sub） --------
    mask_post = (sub["测量时间"] >= meal_datetime) & (sub["测量时间"] <= end_time_180)
    post = sub[mask_post].copy()
    if post.empty:
        G_peak = np.nan
        T_baseline2peak = np.nan
    else:
        idx_peak = post["测量值"].idxmax()
        G_peak = post.loc[idx_peak, "测量值"]
        t_peak_dt = post.loc[idx_peak, "测量时间"]
        T_baseline2peak = (t_peak_dt - meal_datetime).total_seconds() / 60.0  # min

    # -------- 4) 两段斜率 --------
    if pd.isna(T_baseline2peak) or T_baseline2peak == 0:
        S_baseline_peak = np.nan
    else:
        S_baseline_peak = (G_peak - G0) / T_baseline2peak

    G180 = G_points["G180"]
    if pd.isna(T_baseline2peak) or (180 - T_baseline2peak) == 0:
        S_peak_end = np.nan
    else:
        S_peak_end = (G180 - G_peak) / (180 - T_baseline2peak)

    # -------- 5) AUC / pAUC / nAUC / iAUC --------
    t_list = time_points
    g_list = [G0] + [G_points[f"G{t}"] for t in time_points[1:]]

    AUC_0_180 = trapezoid_auc(t_list, g_list)
    pAUC, nAUC, iAUC = compute_p_n_i_auc(G0, t_list, g_list)

    result = {
        "case_id": case_id,
        "qc_status": qc_status,
        "qc_reason": qc_detail.get("reason", ""),
        "meal_date": meal_datetime.date(),
        "meal_datetime": meal_datetime,
        "venous_glu": venous_glu,
        "G0": G0,
        **G_points,
        "G_peak": G_peak,
        "T_baseline2peak_min": T_baseline2peak,
        "S_baseline_peak": S_baseline_peak,
        "S_peak_end": S_peak_end,
        "AUC_0_180": AUC_0_180,
        "pAUC": pAUC,
        "nAUC": nAUC,
        "iAUC": iAUC,
    }

    # 现在返回两个东西：指标 + 原始窗口数据
    return result, raw_window


def qc_cgm_for_meal(df_cgm, meal_datetime, venous_glu=None):
    """
    对某一次馒头餐事件做质控：
    5. 校正数据：比较葡萄糖(Glu)-静脉血与早上6:00-7:00时间段的每个CGM值是否在 Glu±3 范围内
       - CGM 值下限 = 静脉血值 - 3
       - CGM 值上限 = 静脉血值 + 3
    6. 剔除：校准失败数据、餐前30min-餐后210min（早6:30-10:30） CGM数据缺失>30min

    质控结果分为四类：
    - "正常"：校准通过，6:30-10:30 内任意两点间隔 ≤30min，且数据足够计算各指标
    - "校准失败"：6:00-7:00 校正不通过（有点超出 ±3）
    - "数据不够"：6:30-10:30 内存在任意一次间隔 >30min
    - "数据缺失"：馒头餐附近无CGM数据，或者6:00-7:00 完全没有CGM用于校准

    参数
    ----
    df_cgm : DataFrame，至少包含 ['测量时间', '测量值']
    meal_datetime : datetime，本次馒头餐就餐时间（当日早 7:00）
    venous_glu : float 或 None，当天用于校准的静脉血 Glu 值

    返回
    ----
    status : str，"正常" / "校准失败" / "数据不够" / "数据缺失"
    detail : dict，附加信息（可选查看）
    """

    status = "正常"
    detail = {}

    # 如果整天都没 CGM，这肯定是数据缺失
    if df_cgm.empty:
        return "数据缺失", {"reason": "当日无任何CGM记录"}

    # ========= 5. 校正：6:00-7:00 与静脉血比较 =========
    cal_start = datetime.combine(meal_datetime.date(), time(6, 0))
    cal_end   = datetime.combine(meal_datetime.date(), time(7, 0))

    mask_cal = (df_cgm["测量时间"] >= cal_start) & (df_cgm["测量时间"] <= cal_end)
    df_cal = df_cgm[mask_cal].copy()

    if df_cal.empty:
        # 6:00-7:00 完全没有 CGM
        status = "数据缺失"
        detail["reason"] = "6:00-7:00 无CGM用于校准"
        return status, detail

    if venous_glu is not None:
        lower = venous_glu - 3
        upper = venous_glu + 3
        out_of_range = (df_cal["测量值"] < lower) | (df_cal["测量值"] > upper)
        detail["cal_lower"] = lower
        detail["cal_upper"] = upper
        detail["cal_n_points"] = len(df_cal)

        if out_of_range.any():
            status = "校准失败"
            detail["reason"] = "存在CGM值超出静脉血±3范围"

    # ========= 6. 数据完整性：6:30-10:30 是否有缺失>30min =========
    win_start = meal_datetime - timedelta(minutes=30)   # 6:30
    win_end   = meal_datetime + timedelta(minutes=210)  # 10:30

    mask_win = (df_cgm["测量时间"] >= win_start) & (df_cgm["测量时间"] <= win_end)
    df_win = df_cgm[mask_win].copy().sort_values("测量时间")

    if df_win.empty:
        # 馒头餐附近完全没有CGM
        if status == "正常":
            status = "数据缺失"
        detail["reason"] = detail.get("reason", "") + "；6:30-10:30 无CGM记录"
        return status, detail

    # 计算相邻点间隔
    diffs_min = df_win["测量时间"].diff().dt.total_seconds() / 60.0
    max_gap = diffs_min.max()
    detail["max_gap_min"] = float(max_gap) if not np.isnan(max_gap) else None
    detail["n_points_6h30_10h30"] = len(df_win)

    if max_gap is not None and not np.isnan(max_gap) and max_gap > 30:
        # 有任意一次间隔 >30min
        if status == "正常":
            status = "数据不够"
        else:
            # 比如已经是"校准失败"，那就两种问题都有
            status = status + "+数据不够"
        if "reason" in detail:
            detail["reason"] += f"；6:30-10:30 内存在间隔>{max_gap:.1f}min"
        else:
            detail["reason"] = f"6:30-10:30 内存在间隔>{max_gap:.1f}min"

    return status, detail


###############################
# 5. 主流程：遍历前 N 个处理后 CGM 文件
###############################

if __name__ == "__main__":
    # 列出处理后 CGM 目录中的前 N_FILES 个 .xlsx 文件
    pattern = os.path.join(CGM_DIR, "*.xlsx")
    all_cgm_files = sorted(glob.glob(pattern))
    if not all_cgm_files:
        print(f"在目录 {CGM_DIR} 下没有找到任何 .xlsx 文件")
        raise SystemExit(0)

    # 想跑第21到第40个文件：索引从20到39
    start = 0
    end = 20
    cgm_files_to_process = all_cgm_files[start:end]
    print(f"将在目录 {CGM_DIR} 中处理第 {start+1} 到第 {end} 个文件。")

    all_rows = []

    for cgm_path in cgm_files_to_process:
        base_name = os.path.basename(cgm_path)
        name_no_ext = os.path.splitext(base_name)[0]
        # 文件命名可能为 '123456.xlsx' 或 '123456-姓名.xlsx'，只取 '-' 前面的部分作为病案号
        case_str = name_no_ext.split("-")[0]
        try:
            case_id = int(case_str)
        except ValueError:
            case_id = case_str  # 如果不能转成整数，就按字符串匹配

        print(f"\n=== 处理文件: {base_name} (case_id={case_id}) ===")

        # ① 找出该病案号的所有馒头餐就餐时间（当日 7:00）及对应静脉血葡萄糖值
        meal_infos = find_mantou_meals(ORIGINAL_EXCEL_PATH, case_id)

        if not meal_infos:
            print(f"病案号 {case_id} 未检测到任何馒头餐记录，跳过该文件。")
            continue

        print("检测到的馒头餐就餐时间：")
        for info in meal_infos:
            print(f" - {info['meal_datetime']} (Glu={info['venous_glu']})")

        # ② 对每一次馒头餐跑一遍 CGM 指标计算，展开原始CGM窗口的前23个点到同一行
        for info in meal_infos:
            mt = info["meal_datetime"]
            venous_glu = info["venous_glu"]
            res, raw = process_one_meal(cgm_path, case_id, mt, venous_glu=venous_glu)
            if res is not None and raw is not None:
                row = res.copy()
                # 将原始CGM窗口的前 23 个点展开到同一行
                for i in range(23):
                    col_t = f"CGM{i + 1}_time"
                    col_g = f"CGM{i + 1}_value"
                    if i < len(raw):
                        row[col_t] = raw.iloc[i]["测量时间"]
                        row[col_g] = raw.iloc[i]["测量值"]
                    else:
                        row[col_t] = np.nan
                        row[col_g] = np.nan
                all_rows.append(row)

    if not all_rows:
        print("没有任何馒头餐事件成功完成计算，未生成结果表。")
        raise SystemExit(0)

    # ③ 汇总成一个 DataFrame，指定列顺序
    results_df = pd.DataFrame(all_rows)
    base_cols = [
        "case_id", "qc_status", "qc_reason",
        "meal_date", "meal_datetime", "venous_glu",
        "G0", "G15", "G30", "G45", "G60", "G75", "G90", "G105",
        "G120", "G135", "G150", "G165", "G180",
        "G_peak", "T_baseline2peak_min", "S_baseline_peak", "S_peak_end",
        "AUC_0_180", "pAUC", "nAUC", "iAUC",
    ]
    cgm_cols = []
    for i in range(23):
        cgm_cols.append(f"CGM{i + 1}_time")
        cgm_cols.append(f"CGM{i + 1}_value")
    ordered_cols = [c for c in base_cols if c in results_df.columns] + [c for c in cgm_cols if c in results_df.columns]
    results_df = results_df[ordered_cols]

    # ④ 打印结果并写入Excel
    print("\n计算结果预览：")
    print(results_df.to_string(index=False))

    results_df.to_excel("cgm_data.xlsx", index=False)
    print("\n已将结果写入 ./cgm_data.xlsx")