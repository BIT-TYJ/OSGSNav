#!/usr/bin/env python3
"""Create a dependency-free Word summary for the current Multi-IIN work."""

from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUT = Path(__file__).resolve().parents[2] / "Multi-IIN_工作总结与后续计划.docx"


def paragraph(text: str = "", style: str = "Normal") -> str:
    escaped = html.escape(text)
    return (
        '<w:p><w:pPr><w:pStyle w:val="{}"/></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:eastAsia="SimSun"/></w:rPr>'
        '<w:t xml:space="preserve">{}</w:t></w:r></w:p>'
    ).format(style, escaped)


def bullet(text: str) -> str:
    return paragraph(text, "ListBullet")


def build_document() -> str:
    parts = [paragraph("Multi-IIN 工作总结与后续计划", "Title")]
    parts.append(paragraph("记录范围：当前 IEVE 仓库中的 multi_iin_v0 扩展；未修改原始 IEVE 主链，也未提交 Git。"))

    sections = [
        ("一、已完成的修改", [
            "构建 Multi-IIN v0 顺序双目标任务：两个目标固定顺序执行，每个目标 1000 步预算，支持 Independent-Reset、Continuous-Clear、Continuous-Shared。",
            "新增独立 Multi-IIN runner，支持目标切换、目标级计时、成功率/SPL/距离统计，以及 10-episode lifecycle smoke test。",
            "重新生成同楼层 manifest：100 个 episode、5 个 scene、每个 scene 20 个 episode；目标 view-point 与起点高度差不超过 0.75 m。",
            "新增 compact 结果归档，保存目标图像、occupancy/exploration/visited/semantic 合成栅格、最终高斯地图、目标实例点云、渲染图和匹配结果。",
            "新增 Gaussian 容量管理：硬上限、空间 voxel 去冗余、低 opacity 优先裁剪、近期关键帧和目标语义高斯保护。",
            "新增语义实例高斯过滤：voxel + DBSCAN 去杂点、SOIM anchor/确认 seed 选择、同类目标空间关联。",
            "新增类别尺寸一致性门控：bed/sofa/chair/plant/toilet/tv 使用宽松尺度先验；异常小簇会回退到其他簇，没有合理簇时保留 raw 点云。",
            "新增未知区域信息增益优先的 frontier 排序，综合未知区域收益、FMM 代价和已访问惩罚。",
            "新增 FMM 全部不可达时的 frontier recovery：在起点 traversible 连通分量内用 BFS 距离投影和选择目标，不穿墙。",
            "新增 MinAV 策略保护：自适应黑名单半径、更快失败屏蔽、持久拒绝实例、Phase 3 固定步数门控，以及弱证据禁止提前停止。",
            "新增单元测试和 py_compile 验证，覆盖过滤、frontier、策略保护、结果归档和 Gaussian 容量管理。",
        ]),
        ("二、关键实验结论", [
            "第一个 toilet 目标能够成功；第二个 bed 目标失败的主要原因是 detector 反复识别到了错误 bed 实例。",
            "LightGlue 拒绝错误实例，说明匹配门控本身有效；高斯渲染质量是次要因素，不是本次失败的主因。",
            "地图中仍存在 frontier；问题主要是 FMM 将候选判断为不可达，而不是场景已经完全探索。",
            "旧 manifest 没有同楼层约束，约 46/100 个 episode 的目标高度差超过 0.5 m；当前 manifest 已替换为同楼层版本。",
            "详细 MinAV 阶段输出位于旧实验的 minav 目录，compact 是后续主要查看和归档入口。",
        ]),
        ("三、短期目标", [
            "使用新的同楼层 manifest 跑 1 个 Continuous-Shared episode，每个目标 1000 步。",
            "重点检查 target 2 是否真正观察到正确实例、错误实例黑名单是否生效、frontier recovery 是否提供有效探索目标。",
            "确认 Phase 3 严格遵守配置步数，弱匹配证据不能提前停止。",
            "对 bed、sofa、chair、plant、toilet、tv 六类目标做离线过滤复核，检查尺寸门控是否过松或过严。",
            "后续结果优先查看 multi_iin_v0/results/compact/；旧 minav 目录不作为主要归档入口。",
        ]),
        ("四、中期目标", [
            "形成正式 Multi-IIN benchmark：比较 Independent-Reset、Continuous-Clear、Continuous-Shared 三种设置。",
            "报告 Target-1 Success、Target-2 Success、Ordered Success、per-target SPL、总步数、地图覆盖率、高斯数量和跨目标复用收益。",
            "完成最小消融：原始 frontier/信息增益/frontier recovery、原始黑名单/自适应黑名单、无尺寸门控/有尺寸门控、Phase 3 早启用/固定步数启用。",
            "完善同楼层数据集协议，加入楼层标签或高度聚类、目标距离分桶和近/中/远难度划分。",
            "待 Multi-IIN v0 稳定后，再考虑将过滤、frontier recovery 和策略保护以最小改动回灌到原始 IEVE 主链。",
            "围绕有限步数、有限高斯存储和共享场景记忆，整理期刊论文中的方法、消融和系统分析。",
        ]),
    ]
    for heading, items in sections:
        parts.append(paragraph(heading, "Heading1"))
        for item in items:
            parts.append(bullet(item))

    parts.append(paragraph("五、当前重要文件", "Heading1"))
    for item in [
        "同楼层 manifest：multi_iin_v0/manifests/miin_v0_val_k2.json",
        "compact 结果：multi_iin_v0/results/compact/",
        "实例过滤：multi_iin_v0/runtime/semantic_instance_filter.py",
        "frontier 恢复：multi_iin_v0/runtime/frontier_recovery.py",
        "策略保护：multi_iin_v0/runtime/minav_strategy_guards.py",
    ]:
        parts.append(bullet(item))
    return "".join(parts)


def main() -> None:
    document_xml = build_document()
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    document_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
'''
    document = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body>
</w:document>'''.format(document_xml)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
    print(OUT)


if __name__ == "__main__":
    main()
