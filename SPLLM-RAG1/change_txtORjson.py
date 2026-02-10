import os
import fitz  # PyMuPDF
import pandas as pd
import json
import re


# --- 1. 文本清洗辅助函数 ---
def clean_pdf_text(text):
    """
    清洗 PDF 提取出的硬换行，合并被意外切断的句子。
    """
    # 1. 替换连续的多个换行符为特殊标记，保护真正的段落
    text = re.sub(r'\n\s*\n', '[[PARAGRAPH]]', text)

    # 2. 处理单行换行
    lines = text.split('\n')
    cleaned_lines = []

    for i in range(len(lines)):
        line = lines[i].strip()
        if not line:
            continue

        # 启发式判断：如果当前行不是以句号、问号、感叹号、冒号结尾，
        # 说明它很可能和下一行是连接在一起的。
        if not line.endswith(('。', '！', '？', '：', ':', '.', '!', '?')):
            # 如果后面还有行，则去掉换行符（加上空格或直接拼接）
            cleaned_lines.append(line)
        else:
            # 如果是标点结尾，保留换行
            cleaned_lines.append(line + '\n')

    combined_text = "".join(cleaned_lines)

    # 3. 恢复真正的段落
    combined_text = combined_text.replace('[[PARAGRAPH]]', '\n\n')

    # 4. 去除多余的空格（针对英文或排版产生的碎空格）
    combined_text = re.sub(r' +', ' ', combined_text)

    return combined_text


# --- 2. PDF 提取模块 ---
def sync_pdf_to_txt(source_dir, target_dir):
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        return
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]

    for f in pdf_files:
        txt_filename = os.path.splitext(f)[0] + ".txt"
        target_path = os.path.join(target_dir, txt_filename)
        source_path = os.path.join(source_dir, f)

        if not os.path.exists(target_path):
            print(f"🔄 [PDF->TXT] 正在提取并清洗: {f}")
            try:
                doc = fitz.open(source_path)
                raw_text = ""
                for page in doc:
                    raw_text += page.get_text() + "\n"
                doc.close()

                # --- 调用清洗逻辑 ---
                final_text = clean_pdf_text(raw_text)

                with open(target_path, 'w', encoding='utf-8') as tf:
                    tf.write(final_text)
                print(f"✅ 导出成功: {txt_filename}")
            except Exception as e:
                print(f"❌ PDF 转换失败 {f}: {e}")
        else:
            print(f"⏭️ 跳过已存在的 TXT: {f}")


# --- 3. Excel/CSV 转换模块 ---
def sync_excel_to_json(source_dir, target_dir):
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        return
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 包含 csv 后缀
    data_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.xlsx', '.xls', '.csv'))]

    for f in data_files:
        json_filename = os.path.splitext(f)[0] + ".json"
        target_path = os.path.join(target_dir, json_filename)
        source_path = os.path.join(source_dir, f)

        if not os.path.exists(target_path):
            print(f"🔄 [Data->JSON] 正在转换: {f}")
            try:
                ext = os.path.splitext(f)[1].lower()
                all_sheets_data = {}

                if ext == '.csv':
                    # --- 处理 CSV ---
                    try:
                        # 优先尝试 utf-8 (包含带 BOM 的格式)
                        df = pd.read_csv(source_path, encoding='utf-8-sig')
                    except UnicodeDecodeError:
                        # 如果失败，切换到中文常用编码 gbk 或 gb18030
                        df = pd.read_csv(source_path, encoding='gb18030')

                    df = df.fillna("")
                    all_sheets_data["Sheet1"] = df.to_dict(orient='records')

                else:
                    # --- 处理 Excel ---
                    excel_data = pd.read_excel(source_path, sheet_name=None)
                    for sheet_name, df in excel_data.items():
                        df = df.fillna("")
                        all_sheets_data[sheet_name] = df.to_dict(orient='records')

                with open(target_path, 'w', encoding='utf-8') as jf:
                    json.dump(all_sheets_data, jf, ensure_ascii=False, indent=4)
                print(f"✅ 导出成功: {json_filename}")

            except Exception as e:
                print(f"❌ 转换失败 {f}: {e}")
        else:
            print(f"⏭️ 跳过已存在的 JSON: {f}")


# --- 4. 统一执行入口 ---
if __name__ == "__main__":
    print("=== 启动数据预处理流水线 (带文本清洗功能) ===")

    # PDF -> TXT (诊疗规范)
    PDF_SOURCE = "data/pdf_txt"
    PDF_TARGET = "data/MedicalGuide_data"
    sync_pdf_to_txt(PDF_SOURCE, PDF_TARGET)

    print("-" * 30)

    # Excel -> JSON (临床病例)
    EXCEL_SOURCE = "data/excel_json"
    EXCEL_TARGET = "data/ClinicalCase_data"
    sync_excel_to_json(EXCEL_SOURCE, EXCEL_TARGET)

    print("=== 预处理任务已完成 ===")