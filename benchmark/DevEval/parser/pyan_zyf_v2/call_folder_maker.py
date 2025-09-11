"""
    根据call_analyzer.py分析出的结果，构造object各个function间的调用关系
    输出组织成和object原文件夹相同的结构，每个原py文件对应位置放置一个同名的json文件，文件内容为该py文件中的function的调用信息
    调用信息包含以下内容：
        该function的名称
        该function的路径
        该function使用的import语句
        该function调用的其他function路径（按 In-Class，In-File，In-Object 三个层级分组）
"""