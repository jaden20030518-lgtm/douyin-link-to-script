# 抖音链接转品牌脚本

把参考视频的画面文字与口播整理成带角色的原脚本，再由 Codex 按 Skill 中的品牌、产品和优惠规则做最小改写。

本仓库包含实际使用的 Skill、视频下载程序、本地 OCR / ASR、双通道整合程序及历史评测报告。品牌改编由 Codex 执行 `SKILL.md` 中的规则；Python 流水线输出原稿和冲突记录，仓库没有另外实现一个独立的 LLM API 改编服务。

## 源码入口

- [SKILL.md](../SKILL.md)：原稿整理、角色台词及最小品牌改写规则。
- [套餐知识资料](../references/package-map.md)：历史套餐组成、品类、金额与核对项。
- [抖音下载](../scripts/douyin_download.mjs)：通过本机 Chrome 获取视频详情及可用播放地址，保存 MP4。
- [双通道流水线](../helpers/dual_script_pipeline.py)：同时启动 ASR 与 OCR，随后整合原稿并输出运行记录。
- [ASR](../helpers/transcribe_funasr.py)：本地 FunASR Paraformer，包含分段、标点及说话人组件。
- [OCR](../helpers/extract_burned_subs.py)：抽帧并调用 macOS Vision 识别画面文字。
- [整合器](../helpers/merge_asr_subs.py)：按时间对应两路结果，输出原稿、差异和整合结果。
- [领域词表](../helpers/domain_lexicon.py)：历史测试中使用的热词、纠错及杂字过滤规则。

## 环境

OCR 使用 macOS Vision，因此此版本在 macOS 上运行。需要 Python、FFmpeg / ffprobe、Node.js 22 或更新版本，以及 Google Chrome。Node.js 下载程序使用内置 WebSocket。

本次源码验证使用 Python 3.14.0、Node.js 22.23.2。`requirements.txt` 记录本机已经使用的依赖版本；没有验证所有其他系统和 Python 版本的安装组合。首次调用 ASR 可能需要下载模型权重，权重没有放进仓库。

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 运行

### 1. 获取视频

```bash
node scripts/douyin_download.mjs '抖音视频链接' --output-dir ./inputs
```

程序调用本机 Chrome。遇到登录或验证页面，需要在浏览器中完成相应操作。下载文件名会包含视频 ID；下面的 `video.mp4` 指实际下载文件。

### 2. 同时识别画面文字和口播

```bash
./scripts/dual-script ./inputs/video.mp4 --work-root ./work --language zh
```

每条视频的工作目录中会保留：

- `transcripts/`：ASR 文本、词或句子的时间信息及说话人标签。
- `burned_subs.json`、`burned_subs.txt`：OCR 字幕识别结果。
- `burned_subs_clean.json`：轻量清理后的字幕，原始文件仍保留。
- `merge.json`、`diff.json`：整合结果及需要复核的差异。
- `script_original.md`：带角色的原视频脚本。
- `manifest.json`：来源、处理状态及这一识别整合阶段的耗时。

程序对同一视频使用两个线程并行运行 OCR 和 ASR。批量视频默认逐条处理；`--jobs` 控制视频之间的并发，不是 OCR / ASR 是否并行的开关。

```bash
./scripts/dual-script ./inputs/a.mp4 ./inputs/b.mp4 --work-root ./work --jobs 1
./scripts/dual-script --help
```

现有结果默认复用；`--force` 会重做识别并覆盖相应缓存，使用前确认工作目录中的记录是否需要保留。

### 3. 核对原稿，再改写品牌版本

让 Codex 读取此仓库的 `SKILL.md`，并提供视频链接、目标套餐和明确允许替换的字段。先核对 `script_original.md` 与 `diff.json` 中的价格、件数、角色和疑难台词，再保存单独的品牌版本。

对品牌、价格、商品和套餐做受限替换，保留原片的问答顺序、关系钩子、重复和反应。套餐资料为历史口径，使用时提供当前实际活动规则。

若已有逐句核对稿，可将其另存为该视频工作目录里的 `asr_override.txt` 后再次运行流水线；程序会保留整合与差异记录，并将核对稿作为 `script_original.md`。

## 已知限制

OCR 是画面文字识别。字幕区域、卡片文字和遮挡会影响结果；语音也可能误听。现有整合器出现过错误覆盖金额、漏句及角色分配不准，输出需要对照原片核对。

历史比较中 Codex 自主回看原帧、复识别疑难音段的流程，与固定规则整合器是不同的执行策略。公开这份源码不代表已修复报告中列出的全部问题，也不把固定流水线成绩写成自由执行成绩。

完整方法和证据边界见 [2026-09-26 评测报告](../reports/2026-09-26-script-accuracy/README.md)。运行素材、浏览器登录数据、密钥和模型权重不作为源码依赖打包；演示页单独保存核对过的历史示例。

## 本次发布验证

已检查 Python 编译、Node.js 与 Bash 语法和命令帮助；使用历史真实抽帧调用 macOS Vision，读取到“我们是99四件”。另复用同片历史 OCR / ASR 文件运行完整整合入口，确认生成原稿、整合结果、差异记录和 manifest。发布前检查了源码中的凭据与个人绝对路径。

上述整合检查复用了历史识别结果；本次没有重新下载视频或重跑整套 ASR 评测，也没有验证全新机器上的依赖安装。


[回到作品首页](../README.md)
