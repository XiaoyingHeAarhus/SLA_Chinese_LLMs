"""
把 metrics_results.csv 里的每条教师消息导出成单独的 .txt 文件，
供 AlphaReadabilityChinese (ARC) 处理。

文件命名格式：{model}__{level}__{chat_id}__t{turn}.txt
例如：Qwen--Qwen2.5-7B-Instruct__HSK1__20260303-140710__t1.txt

文件名里用 __ 作为分隔符，方便后续合并 ARC 结果时解析回 model/level/chat_id/turn。

用法：
    python export_for_arc.py \
        --input  /path/to/metrics_results_clean.csv \
        --output /path/to/arc_input_folder
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path


def clean_chinese_only(text: str) -> str:
    """
    和 compute_metrics.py 里一样的清理逻辑。
    ARC 只需要中文内容，去掉英文/拼音/emoji 减少干扰。
    """
    # 删除括号内的拼音/英文注释
    text = re.sub(r'\([^)]*[a-zA-Zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ][^)]*\)', '', text)
    # 删除英文单词
    text = re.sub(r'[a-zA-Z]+', '', text)
    # 删除声调拼音
    text = re.sub(r'[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+\w*', '', text)
    # 删除 emoji 和非中文 Unicode 符号
    text = re.sub(r'[^\u0000-\u4DFF\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF\n]', '', text)
    # 合并多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def safe_filename(model: str, level: str, chat_id: str, turn: int) -> str:
    """
    生成安全的文件名，替换掉文件系统不允许的字符。
    用 __ 作为字段分隔符（单下划线可能出现在 model 名里）。
    """
    # 替换文件名中不允许的字符
    model_safe = re.sub(r'[\\/:*?"<>|]', '-', model)
    return f"{model_safe}__{level}__{chat_id}__t{turn}.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="metrics_results_clean.csv 的路径")
    parser.add_argument("--output", required=True,
                        help="导出 txt 文件的目标文件夹")
    parser.add_argument("--no-clean", action="store_true",
                        help="不清理文本，直接导出原始内容（默认会清理）")
    args = parser.parse_args()

    # 读取 CSV
    df = pd.read_csv(args.input, encoding="utf-8")

    # 过滤掉 .ipynb_checkpoints 污染（以防用了未清理的原始CSV）
    df = df[df["model"] != "v4.0"]
    df = df[df["level"] != ".ipynb_checkpoints"]
    df = df[~df["chat_id"].str.contains("checkpoint", na=False)]

    # 创建输出文件夹
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    exported = 0

    for _, row in df.iterrows():
        content = str(row["content"])

        # 清理文本（默认开启）
        if not args.no_clean:
            content = clean_chinese_only(content)

        # 跳过清理后近空的消息
        if len(content.strip()) < 2:
            skipped += 1
            continue

        # 生成文件名
        fname = safe_filename(
            model   = str(row["model"]),
            level   = str(row["level"]),
            chat_id = str(row["chat_id"]),
            turn    = int(row["turn"])
        )

        # 写入 txt（UTF-8 编码，ARC 要求）
        outpath = output_dir / fname
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(content)

        exported += 1

    print(f"导出完成：{exported} 个文件 → {output_dir}")
    if skipped:
        print(f"跳过近空消息：{skipped} 条")
    print(f"\n文件命名格式：{{model}}__{{level}}__{{chat_id}}__t{{turn}}.txt")
    print("ARC 跑完后，用 merge_arc_results.py 把结果合并回 CSV。")


if __name__ == "__main__":
    main()
