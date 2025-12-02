import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Any

# 原始文件路径与工作表
EXCEL_PATH = "4_计算新特征后_1340.xlsx"
SHEET_NAME = "检验"

# 输出文件路径
OUTPUT_PATH = "merged_mantou_features.xlsx"

# 需要汇总的列（按 Excel 列字母）
TARGET_COL_LETTERS = [
    "I", "J", "K", "L", "N", "P", "Q", "R", "S", "T", "V", "Y", "Z",
    "AD", "AF", "AI", "AK", "AM", "AP", "AR", "AS", "AT", "AU", "AV",
    "AW", "AY", "AZ", "BC", "BF", "BG", "BH", "BI", "BJ"
]


def excel_col_to_idx(col: str) -> int:
    """
    将 Excel 列字母转换为 0 基索引（A -> 0, B -> 1, ..., Z -> 25, AA -> 26, ...）。
    """
    col = col.upper()
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1  # 转为 0-based


def main():
    # 1. 读入检验表
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # 确保时间列能正确转为 datetime
    # D 列 = 病案号，H 列 = 时间，BC 列 = 触发列
    col_case = df.columns[excel_col_to_idx("D")]
    col_time = df.columns[excel_col_to_idx("H")]
    col_trigger = df.columns[excel_col_to_idx("BC")]

    # 记录列 T 和 BF 对应的列名，用于后续特殊处理
    col_T_name = df.columns[excel_col_to_idx("T")]
    col_BF_name = df.columns[excel_col_to_idx("BF")]

    # 将时间列转为 datetime，并单独抽出“日期”列
    df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
    df["__DATE__"] = df[col_time].dt.date

    # 2. 准备目标列名（用 Excel 列字母映射到真正的表头）
    target_col_names: List[str] = []
    for letter in TARGET_COL_LETTERS:
        idx = excel_col_to_idx(letter)
        if idx < len(df.columns):
            target_col_names.append(df.columns[idx])
        else:
            raise ValueError(f"列 {letter} 超出当前表的列范围，请检查。")

    # 3. 找出所有“触发行”：BC 列有值的行
    mask_trigger = df[col_trigger].notna()
    trigger_rows = df[mask_trigger].copy()

    # 从触发行中提取所有 (病案号, 馒头餐日期) 组合
    # 注意馒头餐日期使用 H 列的日期部分
    trigger_rows = trigger_rows[[col_case, "__DATE__"]].dropna()
    trigger_rows = trigger_rows.drop_duplicates()

    # key: (case_id, date) -> 只处理一次
    events: List[Tuple[Any, Any]] = list(
        trigger_rows.itertuples(index=False, name=None)
    )

    merged_rows: List[Dict[str, Any]] = []

    for case_id, meal_date in events:
        # 4. 找出该病案号在该日期的所有行
        sub = df[(df[col_case] == case_id) & (df["__DATE__"] == meal_date)]

        if sub.empty:
            # 理论上不应发生，保险打印一下
            print(f"警告：病案号 {case_id} 在 {meal_date} 没有匹配到任何行。")
            continue

        row_out: Dict[str, Any] = {}
        row_out["住院号"] = case_id
        row_out["馒头餐日期"] = meal_date

        conflict_cols: List[str] = []

        # 5. 对每一个目标列做汇总
        for col_name in target_col_names:
            col_series = sub[col_name].dropna()
            # 没有检测到任何值
            if col_series.empty:
                row_out[col_name] = pd.NA
                continue

            # 所有唯一非空值
            values = col_series.unique()

            # 只有一个唯一值：直接使用
            if len(values) == 1:
                row_out[col_name] = values[0]
                continue

            # 存在多个不同的值：冲突情况
            # 对于 T 列，按特殊规则处理：取 BF 列不为空的那行的 T 值，并不计作冲突
            if col_name == col_T_name:
                # 在当前子表中，找 BF 列非空的行
                sub_with_bf = sub[sub[col_BF_name].notna()]
                if not sub_with_bf.empty:
                    # 从这些行中取对应 T 列的值
                    t_values = sub_with_bf[col_name].dropna().unique()
                    if len(t_values) >= 1:
                        # 使用 BF 非空行对应的 T 值，不计冲突
                        chosen = t_values[0]
                        row_out[col_name] = chosen
                        # 可选：在终端提示一下
                        print(
                            f"T 列多值，按 BF 非空行取值：病案号 {case_id}, 日期 {meal_date}, 列 {col_name} 使用值 {chosen}"
                        )
                        continue
                # 如果没有找到 BF 非空行，或该行 T 也为空，则退回普通冲突逻辑

            # 普通列（或 T 列无法按 BF 非空行判定）——按原先逻辑：取第一个值，并记录冲突
            row_out[col_name] = values[0]
            conflict_cols.append(col_name)
            print(
                f"冲突：病案号 {case_id}, 日期 {meal_date}, 列 {col_name} 存在多个值 {values}"
            )

        # 记录冲突列名（如果有）
        if conflict_cols:
            row_out["冲突列"] = ",".join(conflict_cols)
        else:
            row_out["冲突列"] = ""

        merged_rows.append(row_out)

    # 6. 整理成 DataFrame 并输出 Excel
    if not merged_rows:
        print("没有生成任何馒头餐事件的合并结果。")
        return

    merged_df = pd.DataFrame(merged_rows)

    # 调整列顺序：住院号、馒头餐日期、冲突列、其余特征列
    cols_order = ["住院号", "馒头餐日期", "冲突列"] + target_col_names
    # 只保留存在的列（防止意外）
    cols_order = [c for c in cols_order if c in merged_df.columns]
    merged_df = merged_df[cols_order]

    # 写出到新的 Excel 文件
    merged_df.to_excel(OUTPUT_PATH, index=False)
    print(f"已将合并结果写入：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()