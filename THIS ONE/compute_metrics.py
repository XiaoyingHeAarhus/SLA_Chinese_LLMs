"""
中文HSK对齐漂移复现研究的指标计算脚本。
尽量贴近 Almasi & Kristensen-McLachlan (2025) 的原始方法。

计算的四个指标（metrics）：
  1. 句子长度（Sentence Length）     — 替代原文的西班牙语可读性公式
  2. 文本长度（Text Length）         — 与原文相同，token数量
  3. 平均依存距离（MDD）              — 与原文相同，衡量句法复杂度
  4. 信息惊异度（Message Surprisal） — 与原文方法相同，换用中文模型

依赖版本要求（version requirements）：
  transformers==4.47.0  ← 必须是4.x，5.x移除了batch_encode_plus，minicons会报错

用法（usage）：
    python compute_metrics.py --data_dir /path/to/json/files --output results.csv
"""

# ── 导入模块（Imports） ────────────────────────────────────────────────────────
# 每行导入一个"工具箱"，后面的代码会用到它们

import os           # 操作系统（operating system）工具，处理文件路径等
import re           # 正则表达式（regular expressions），用于文本模式匹配和替换
import json         # 读写JSON文件，你的对话数据就是以JSON格式存储的
import argparse     # 解析命令行参数（command-line arguments），比如 --data_dir
import numpy as np  # 数值计算库（numerical python），主要用来求平均值（mean）
import pandas as pd # 表格数据处理库，相当于Python版的Excel
from pathlib import Path  # 跨操作系统的文件路径工具，比字符串拼接路径更安全


# ── 第一部分：辅助函数（Helper Functions） ────────────────────────────────────
# 这些小函数会被后面的指标计算函数反复调用

def extract_tutor_messages(dialogue: list) -> list[dict]:
    """
    为什么需要这个函数：
    我们只分析教师（tutor/assistant）的消息，不分析学生的。
    原文第5节写道："We focus solely on analyzing the tutor LLM's responses"。

    怎么做：
    遍历对话列表，每条消息都有一个 "role"（角色）字段，
    值为 "system"（系统提示）、"user"（学生）或 "assistant"（教师）。
    只收集 assistant 的消息，并给每条消息编一个轮次（turn）编号。
    """
    messages = []  # 用空列表收集结果
    turn = 0       # 轮次计数器，从0开始，第一条教师消息会变成turn=1

    for msg in dialogue:
        # msg 是一条消息，例如 {"role": "assistant", "content": "你好！..."}
        if msg["role"] == "assistant":   # 只处理教师消息，跳过系统提示和学生消息
            turn += 1                    # 计数器递增：0→1, 1→2，以此类推
            messages.append({
                "turn": turn,             # 轮次编号，1到9，对应论文图表的x轴
                "content": msg["content"] # 消息的实际文本内容
            })

    return messages  # 返回收集好的教师消息列表


def clean_chinese_only(text: str) -> str:
    """
    为什么需要这个函数：
    你的数据混杂了多种内容——拼音（Pinyin）如"(Nǐ hǎo)"、英文短语、汉字
    都可能出现在同一条消息里。如果不清理，指标就会同时测量英文和拼音，
    从而扭曲结果。原文用 lingua 库过滤整条非西班牙语消息；因为你的数据
    是在消息内部混用语言，所以我们在原文本内直接做行内清理（inline cleaning）。

    还包括 emoji 去除：
    原文第5节明确写道 "the only preprocessing applied was the removal of emojis
    from Gemma's outputs"。我们对所有模型统一处理，因为实际上只有Gemma会输出emoji。

    怎么做：
    用正则表达式（regular expressions）逐步删除拼音、英文和emoji，
    只保留汉字和中文标点。
    """

    # 第一步：删除括号内含有拉丁字母或声调字母的内容
    # 比如 (Nǐ hǎo)、(Hello!) 这类括号注释
    # 正则模式解释：
    #   \(          = 左括号
    #   [^)]*       = 括号内任意字符（不包含右括号）
    #   [a-zA-Z...] = 至少一个拉丁字母（含声调字母）
    #   [^)]*       = 更多括号内字符
    #   \)          = 右括号
    text = re.sub(r'\([^)]*[a-zA-Zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ][^)]*\)', '', text)

    # 第二步：删除括号外剩余的英文单词（连续的a-z或A-Z字符串）
    text = re.sub(r'[a-zA-Z]+', '', text)

    # 第三步：删除括号外残留的声调拼音字母
    # 比如 "āáǎà" 是拼音字母"a"的四个声调形式
    text = re.sub(r'[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+\w*', '', text)

    # 第四步：删除 emoji 和其他 Unicode 特殊符号
    # 只保留以下 Unicode 范围：
    #   \u0000-\u4DFF  = 基本拉丁字符、数字、常用标点等
    #   \u4E00-\u9FFF  = CJK统一汉字（标准中文字符区）
    #   \u3000-\u303F  = CJK标点（。！？「」等）
    #   \uFF00-\uFFEF  = 全角字符（，、；：等）
    #   \n             = 换行符（保留，用于后续断句）
    # 所有不在这些范围内的字符（包括emoji）都会被删除
    text = re.sub(r'[^\u0000-\u4DFF\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF\n]', '', text)

    # 第五步：合并多余空白（whitespace），去除首尾空格
    # \s+ 表示"一个或多个空白字符"，统一替换成单个空格
    text = re.sub(r'\s+', ' ', text).strip()  # .strip() 删除首尾空格

    return text


def split_sentences_zh(text: str) -> list[str]:
    """
    为什么需要这个函数：
    句子长度（sentence length）和信息惊异度（message surprisal）都是先在
    句子层面计算，再取平均。所以需要先把消息切分成单个句子。
    这是原文西班牙语版按 ". ! ?" 断句的中文等价做法。

    怎么做：
    按中文句末标点断句：。（句号）！（感叹号）？（问号）和换行符。
    然后过滤掉空字符串和单字符片段（通常是标点符号残留）。
    """

    # re.split 在每个匹配位置切分文本
    # [。！？\n]+ 表示"一个或多个这类字符"（允许连续标点）
    sentences = re.split(r'[。！？\n]+', text)

    # 只保留长度大于1的句子（过滤空字符串和单个标点残留）
    sentences = [s.strip() for s in sentences if len(s.strip()) > 1]

    return sentences


# ── 第二部分：指标1 — 句子长度（Sentence Length） ─────────────────────────────
# 替代原文三个西班牙语可读性公式（Fernández Huerta、Szigriszt-Pazos、
# Gutiérrez de Polini），这三个公式都依赖音节数（syllable count），
# 而中文没有音节数这个概念，无法直接套用。

def compute_sentence_length(text: str) -> float:
    """
    为什么这样替代：
    原文三个可读性指标本质上都在测量"每词音节数"和"每句词数"。
    中文没有对应的音节可读性公式，所以用平均句子字符数作为代替。
    句子越长通常代表文本越复杂。

    方向注意（direction note）：
    原文西班牙语指标是分数越高越简单（高分=易读）。
    我们的句子长度是越高越难（高分=句子越长=越复杂）。
    因此预期模式应该是 HSK1 < HSK3 < HSK5（而不是反过来）。

    怎么做：
    清理文本 → 断句 → 统计每句字符数 → 取平均。
    """

    cleaned = clean_chinese_only(text)       # 先去除拼音和英文
    sentences = split_sentences_zh(cleaned)  # 切分成句子列表

    if not sentences:   # 如果清理后什么都没剩（比如消息全是emoji）
        return np.nan   # np.nan 表示"非数字"（not a number），是安全的缺失值标记

    # 对每个句子，统计非空白字符（non-whitespace characters）的数量
    # re.sub(r'\s', '', s) 先删除句子内的空格，再用 len() 统计字符数
    char_counts = [len(re.sub(r'\s', '', s)) for s in sentences]

    # np.mean() 计算列表的平均值
    # float() 把 numpy 浮点数转成标准 Python 浮点数，CSV输出更干净
    return float(np.mean(char_counts))


# ── 第三部分：指标2和3 — 文本长度与MDD（Text Length & MDD） ──────────────────
# 这两个指标直接来自原文第4.2节"结构复杂度（Structural Complexity）"。
# 原文用 TextDescriptives 库计算，但该库不支持中文，
# 所以我们用 spaCy 手动复现同样的计算逻辑。

def load_spacy_model():
    """
    为什么需要这个函数：
    spaCy 是自然语言处理（NLP, Natural Language Processing）库。
    它的中文模型 zh_core_web_md 能对中文进行分词（tokenization）
    和依存句法分析（dependency parsing）。

    我们在脚本开始时只加载一次，然后传给每个需要它的函数复用，
    因为加载模型很慢，不能每条消息都重新加载（3240次会非常低效）。

    zh_core_web_md 对应原文使用的西班牙语模型 es_core_news_md。
    """
    try:
        import spacy
        nlp = spacy.load("zh_core_web_md")  # 加载中文中型模型
        return nlp
    except OSError:
        # OSError 说明模型还没下载，给出提示
        raise OSError(
            "找不到中文 spaCy 模型，请先安装：\n"
            "  python -m spacy download zh_core_web_md"
        )


def compute_structural_metrics(text: str, nlp) -> dict:
    """
    在同一个 spaCy 解析结果（doc）上同时计算两个指标，避免重复解析。

    指标2 — 文本长度（Text Length）：
    为什么：原文把 token 数量作为结构特征之一。HSK级别越高，教师消息
    通常越长。这也是原文图4中对齐漂移（alignment drift）最明显的指标之一。
    怎么做：统计 spaCy 识别的 token 总数，排除纯空白 token。

    指标3 — 平均依存距离（MDD, Mean Dependency Distance）：
    为什么：MDD 衡量句法复杂度（syntactic complexity）。"依存关系
    （dependency）"是两个词之间的语法关系——比如"我喜欢历史"中，
    "我"和"历史"都依存于"喜欢"。两个词在句子中相距越远，说明
    句子结构越复杂。原文采用 Oya (2011) 的 MDD 定义。
    怎么做：对每个词计算 |该词位置 - 其语法中心词位置|，
    在句子内取平均，再对所有句子取平均。
    """

    cleaned = clean_chinese_only(text)  # 先去除拼音、英文和emoji

    # 如果清理后文本为空（比如消息全是英文），
    # 返回 NaN 而不是让程序崩溃
    if not cleaned.strip():
        return {"text_length": np.nan, "mdd": np.nan}

    # nlp(cleaned) 运行完整的 spaCy 处理流水线（pipeline）：
    # 分词（tokenization）→ 词性标注（POS tagging）→ 依存解析（dependency parsing）
    # 结果 doc 包含所有语言学标注信息
    doc = nlp(cleaned)

    # ── 文本长度（Text Length） ───────────────────────────────────────────────
    # 统计所有非空白 token 的数量
    # t.is_space 为 True 表示该 token 是纯空白字符（空格、换行等）
    text_length = len([t for t in doc if not t.is_space])

    # ── 平均依存距离（MDD） ───────────────────────────────────────────────────
    sentence_mdds = []  # 收集每个句子的MDD值

    for sent in doc.sents:      # doc.sents 遍历 spaCy 识别的每个句子
        tokens = list(sent)

        if len(tokens) < 2:     # 单词句子无法计算距离，跳过
            continue

        distances = []  # 收集当前句子内所有词的依存距离

        for token in tokens:
            # token.head 是该 token 的语法中心词（grammatical head）
            # 如果 token.head == token，说明该词是句子的根节点（root），
            # 根节点没有中心词，跳过以避免引入距离为0的干扰
            if token.head != token:
                # token.i 是该词在整个文档中的位置索引
                # token.head.i 是其中心词的位置索引
                # abs() 取绝对值，距离永远为正数
                dist = abs(token.i - token.head.i)
                distances.append(dist)

        if distances:   # 如果找到至少一个依存距离
            sentence_mdds.append(np.mean(distances))  # 记录该句子的平均距离

    # 对所有句子的MDD取平均，得到消息级别的MDD
    mdd = float(np.mean(sentence_mdds)) if sentence_mdds else np.nan

    return {"text_length": text_length, "mdd": mdd}


# ── 第四部分：指标4 — 信息惊异度（Message Surprisal） ────────────────────────
# 直接遵循原文第4.3节的方法。
# 唯一改动：把针对欧洲语言的 EuroBERT 替换为中文等价模型
# hfl/chinese-roberta-wwm-ext。
#
# 版本兼容性警告（version compatibility warning）：
# 需要 transformers==4.47.0（4.x版本），transformers 5.x 移除了
# batch_encode_plus 方法，会导致 minicons 报 AttributeError。

def load_surprisal_model(device="cpu"):
    """
    为什么需要这个函数：
    我们需要一个 BERT 类语言模型来计算惊异度分数（surprisal scores）。
    minicons 是一个专门从 BERT 模型中提取惊异度的 Python 库。
    和 spaCy 模型一样，只在脚本开始时加载一次，然后复用。

    hfl/chinese-roberta-wwm-ext 由哈工大讯飞联合实验室（HFL）开发，
    是中文NLP领域最广泛使用的 BERT 模型，是原文 EuroBERT 的中文等价物。

    device="cpu"  表示用CPU运算（较慢，但所有机器都有）
    device="cuda" 表示用GPU运算（快很多，在UCloud上请用这个）
    """
    from minicons import scorer  # 在函数内导入，避免拖慢脚本启动速度

    model = scorer.MaskedLMScorer(
        "hfl/chinese-roberta-wwm-ext",  # 从 HuggingFace 加载此模型
        device=device                    # 指定运算设备
    )
    return model


def compute_message_surprisal(text: str, model) -> float:
    """
    为什么需要这个指标：
    惊异度（surprisal）衡量一个词在其上下文中出现的"出乎意料程度"。
    复杂、高级的文本倾向于使用更难预测的词汇，因此更高的HSK级别
    应对应更高的惊异度。原文发现该指标有预期的排序（A1>B1>C1的惊异度），
    但也是六个指标中最弱、最不稳定的一个。

    怎么做（严格遵循原文第4.3节）：
      1. 把消息切分成句子
      2. 对每个句子计算 token 级别的平均惊异度
         （= 负对数概率，按 token 数量归一化）
      3. 对所有句子的惊异度取平均 → 消息惊异度（Message Surprisal）

    PLL_metric='within_word_l2r'：
    这是掩码语言模型（masked language model）的评分方式。
    'within_word_l2r' 是 Kauf & Tily (2023) 提出的改进版本，
    比原始掩码评分更准确。原文未指定具体变体，此为当前推荐默认值。
    """

    cleaned = clean_chinese_only(text)       # 去除拼音、英文和emoji
    sentences = split_sentences_zh(cleaned)  # 切分成句子列表

    if not sentences:    # 清理后什么都没剩
        return np.nan

    try:
        # sequence_score 对列表中每个句子分别计算一个分数
        # reduction=lambda x: -x.mean(0).item() 的含义：
        #   x        = 句子中每个 token 的对数概率（log-probability）张量（tensor）
        #   -x       = 取反（对数概率是负数，惊异度是正数）
        #   .mean(0) = 对所有 token 取平均（按句子长度归一化）
        #   .item()  = 把 PyTorch 张量转成普通 Python 浮点数
        surprisals = model.sequence_score(
            sentences,
            reduction=lambda x: -x.mean(0).item(),
            PLL_metric='within_word_l2r'
        )
        # surprisals 现在是每个句子对应一个浮点数的列表
        # np.mean() 对所有句子的惊异度取平均，得到消息级别的惊异度
        return float(np.mean(surprisals))

    except Exception as e:
        # 如果某条消息出现问题（比如句子太长超过模型上限），
        # 打印警告信息，返回 NaN，而不是让整个脚本崩溃
        print(f"  警告：惊异度计算失败，文本片段：{text[:50]!r} — {e}")
        return np.nan


# ── 第五部分：文件处理（File Processing） ────────────────────────────────────

def process_file(
    filepath: str,
    nlp,
    surprisal_model,
    model_name: str,
    level: str,
    chat_id: str
) -> list[dict]:
    """
    为什么需要这个函数：
    这个函数端到端地处理一个 JSON 文件（= 一段对话）。
    读取文件 → 提取教师消息 → 对每条消息计算四个指标
    → 返回结果行列表（可直接添加到输出表格）。
    """

    # open() 打开文件；"r" 表示只读；encoding="utf-8" 确保能正确读取中文
    with open(filepath, "r", encoding="utf-8") as f:
        dialogue = json.load(f)  # json.load() 把 JSON 文本解析成 Python 列表

    tutor_messages = extract_tutor_messages(dialogue)  # 只提取教师轮次
    rows = []  # 收集当前对话的所有结果行

    for msg in tutor_messages:
        turn    = msg["turn"]     # 轮次编号（1–9）
        content = msg["content"]  # 消息的原始文本

        # 跳过清理后实质上为空的消息
        # 比如消息只包含 emoji（Gemma的已知问题），清理后就变成空字符串
        # len(...) < 2 表示清理后剩余字符数不足2个
        if len(clean_chinese_only(content).strip()) < 2:
            print(f"  跳过近空消息 turn {turn} in {chat_id}")
            continue  # continue 跳过本次循环的剩余代码，直接处理下一条消息

        # 计算四个指标
        sent_len  = compute_sentence_length(content)
        struct    = compute_structural_metrics(content, nlp)
        surprisal = compute_message_surprisal(content, surprisal_model)

        # 把结果整理成一个字典（dictionary），对应输出CSV的一行
        rows.append({
            "model":             model_name,            # 例如 "Qwen--Qwen2.5-7B-Instruct"
            "level":             level,                 # 例如 "HSK1"、"HSK3"、"HSK5"
            "chat_id":           chat_id,               # 例如 "20260303-162952"（每段对话唯一标识）
            "turn":              turn,                  # 1–9（论文图表的x轴）
            "content":           content,               # 原始文本（方便后续人工核查）
            "sentence_length":   sent_len,              # 指标1：平均句子字符数
            "text_length":       struct["text_length"], # 指标2：token总数
            "mdd":               struct["mdd"],         # 指标3：平均依存距离
            "message_surprisal": surprisal,             # 指标4：消息惊异度
        })

    return rows  # 返回当前对话的所有结果行


# ── 第六部分：主函数（Main Function） ────────────────────────────────────────
# 程序的"入口点（entry point）"，从终端运行脚本时会执行这个函数。
# 它把所有部分串联起来。

def main():

    # argparse 让我们从终端传入参数，例如：
    #   python compute_metrics.py --data_dir /work/... --output results.csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="包含所有JSON对话文件的根目录")
    parser.add_argument("--output",   default="metrics_results.csv",
                        help="输出CSV文件的保存路径")
    parser.add_argument("--device",   default="cpu",
                        help="惊异度模型的运算设备：'cpu' 或 'cuda'（有GPU时用cuda）")
    args = parser.parse_args()
    # 执行完这行后，args.data_dir、args.output、args.device 分别保存你传入的值

    # 在脚本开始时只加载一次模型（加载很慢，每次约30秒）
    # 加载好之后传给每个需要它的函数，避免重复加载
    print("正在加载 spaCy 中文模型...")
    nlp = load_spacy_model()

    print("正在加载惊异度模型（hfl/chinese-roberta-wwm-ext）...")
    surprisal_model = load_surprisal_model(device=args.device)

    all_rows = []  # 收集所有文件、所有对话的结果行

    # Path() 把字符串转成 Path 对象，提供 .rglob() 等实用方法
    data_dir   = Path(args.data_dir)

    # .rglob("*.json") 递归（recursively）搜索所有子文件夹中的JSON文件
    # sorted() 让处理顺序在每次运行时保持一致，方便调试
    json_files = sorted(data_dir.rglob("*.json"))
    print(f"找到 {len(json_files)} 个JSON文件。")

    for i, filepath in enumerate(json_files):
        # enumerate() 同时给出索引 i（0, 1, 2...）和文件路径 filepath
        # i+1 使显示从"1/360"开始而不是"0/360"
        print(f"[{i+1}/{len(json_files)}] 处理中：{filepath.name}...")

        # 从文件夹结构中提取模型名和HSK级别：
        # .../simulated_data/Qwen--Qwen2.5-7B-Instruct/v4.0/HSK3/20260303-162952.json
        #                    ↑ parts[-4]（模型名）             ↑ parts[-2]（HSK级别）
        parts      = filepath.parts   # 把完整路径拆分成各级文件夹名的元组（tuple）
        model_name = parts[-4]        # 倒数第4个 = 模型名文件夹
        level      = parts[-2]        # 倒数第2个 = HSK级别文件夹
        chat_id    = filepath.stem    # .stem 去掉".json"后缀，得到文件名主体

        rows = process_file(
            filepath=str(filepath),
            nlp=nlp,
            surprisal_model=surprisal_model,
            model_name=model_name,
            level=level,
            chat_id=chat_id
        )
        all_rows.extend(rows)  # .extend() 把 rows 中的每一项都追加到 all_rows

    # 把所有结果行的列表转成 pandas DataFrame（数据表格）
    df = pd.DataFrame(all_rows)

    # 保存为 CSV 文件，之后在 R 中用于混合效应模型（linear mixed effects models）
    # index=False 防止 pandas 多加一列行号
    df.to_csv(args.output, index=False)
    print(f"\n完成！结果已保存至 {args.output}")

    # 打印快速汇总表：每个模型 × HSK级别的各指标均值
    # 用来立即检验是否出现 HSK1 < HSK3 < HSK5 的预期模式
    print("\n各模型 × HSK级别的指标均值：")
    print(
        df.groupby(["model", "level"])[
            ["sentence_length", "text_length", "mdd", "message_surprisal"]
        ].mean().round(3)
    )


# ── 脚本入口（Script Entry Point） ───────────────────────────────────────────
# 这行的含义：只有当这个文件被直接从终端运行时才执行 main()，
# 如果被其他 Python 脚本作为模块导入（import），则不执行
if __name__ == "__main__":
    main()
