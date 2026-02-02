from __future__ import annotations

import argparse
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]


def seed_kb(kb_root: Path) -> None:
    kb_root = kb_root.resolve()

    _write(
        kb_root / "hospital" / "sop_intake.md",
        """
# 门诊通用SOP：问诊要点与分流（示例）

## 问诊要点
- 主诉与时间线：起病时间、持续时间、伴随症状
- 既往史：慢病、手术史、住院史
- 用药史：处方药、OTC、保健品
- 过敏史：药物/食物/其他
- 家族史与社会史：吸烟饮酒、职业暴露

## 分流/红旗（示例）
- 黑便/呕血/大量便血：建议急诊评估
- 意识障碍、突发偏瘫/言语不清：建议急诊评估

## 免责声明
本资料仅用于流程仿真与宣教模板示例，不构成医疗建议。
""",
    )

    _write(
        kb_root / "hospital" / "sop_billing_reports.md",
        """
# 门诊通用SOP：缴费/预约/取报告/回诊（示例）

## 缴费
- 按医院指引完成线上或线下缴费；缴费后方可进行检查/检验。

## 预约与准备
- 需要预约的检查（如内镜、MRI、EEG）请按预约时间到院。
- 若需禁食/停药/肠道准备等，请严格按准备清单执行。

## 取报告与回诊
- 检查完成后可在自助机/APP/窗口查询并领取报告。
- 持报告返回诊室复诊，由医生综合分析并制定方案。
""",
    )

    _write(
        kb_root / "hospital" / "sop_followup.md",
        """
# 门诊通用SOP：诊后处置与随访（示例）

## 诊后处置
- 门诊取药、治疗或进一步检查预约
- 如触发红旗/升级：急诊/住院/会诊/转诊

## 随访
- 按医嘱复诊时间复诊
- 症状加重或出现红旗时立即线下就医/急诊
""",
    )

    _write(
        kb_root / "hospital" / "education_common.md",
        """
# 通用宣教与随访要点（示例）

## 通用宣教
- 规范用药：按医嘱使用，不随意增减
- 记录症状：发作时间、诱因、缓解因素
- 生活方式：规律作息、避免过量酒精

## 红旗/应急
- 意识障碍、突发偏瘫/言语不清
出现上述情况请立即线下就医/急诊。
""",
    )

    _write(
        kb_root / "neuro" / "guide_redflags.md",
        """
# 神经内科：红旗与检查选择（示例）

## 红旗
- 突发偏瘫/言语不清/意识障碍：卒中待排，建议急诊
- 反复抽搐/癫痫持续：建议急诊

## 常用检查
- 影像：CT/MRI/CTA
- 电生理：EEG、EMG/NCV
- 实验室：血常规/生化等（必要时 CSF）
""",
    )

    _write(
        kb_root / "neuro" / "prep_mri.md",
        """
# MRI/CTA 注意事项（示例）

- 体内金属植入物需评估禁忌
- CTA 需评估造影剂过敏与肾功能
- 依医院流程预约并按时到院
""",
    )

    _write(
        kb_root / "neuro" / "prep_eeg.md",
        """
# EEG 注意事项（示例）

- 检查前避免咖啡因与熬夜（以医嘱为准）
- 依预约时间到院
""",
    )

    _write(
        kb_root / "neuro" / "plan_neuro.md",
        """
# plan：神经内科常见症状处理与长期管理模板（示例）

- 头痛/眩晕：对症治疗 + 诱因管理 + 记录发作；出现红旗及时就医
- 癫痫样发作：完善 EEG 与影像，遵医嘱长期管理，避免危险作业
- 卒中红旗：急诊评估、影像优先、必要时住院
""",
    )

    _write(
        kb_root / "neuro" / "education_neuro.md",
        """
# 神经内科宣教要点（示例）

- 监测：记录症状频率与诱因
- 应急：意识障碍/偏瘫/言语不清/持续抽搐立即急诊
- 长期：按医嘱用药与复诊，设定长期管理目标
""",
    )

    _write(
        kb_root / "forms" / "template_emr.md",
        """
# 表单模板：门诊病历（示例）

字段示例：主诉、现病史、既往史、过敏史、查体、初步诊断、处理计划、随访。
""",
    )
    _write(
        kb_root / "forms" / "template_diagnosis_cert.md",
        """
# 表单模板：诊断证明（示例）

字段示例：姓名、就诊日期、诊断、建议、盖章。
""",
    )
    _write(
        kb_root / "forms" / "template_sick_leave.md",
        """
# 表单模板：病假条（示例）

字段示例：姓名、休假天数、起止日期、医嘱。
""",
    )
    _write(
        kb_root / "forms" / "template_education_sheet.md",
        """
# 表单模板：宣教单（示例）

字段示例：教育要点、随访计划、红旗/应急处理、免责声明。
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed minimal KB examples under ./kb")
    parser.add_argument(
        "--kb",
        type=Path,
        default=ROOT / "kb",
        help="KB root directory (default: <repo>/kb)",
    )
    args = parser.parse_args()
    kb_root = Path(args.kb)
    if not kb_root.is_absolute():
        kb_root = (ROOT / kb_root).resolve()
    seed_kb(kb_root)
    print(f"Seeded KB under: {args.kb.resolve()}")


if __name__ == "__main__":
    main()
