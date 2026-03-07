# AWB 智能自动白平衡工具 (AWB Pro)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-orange.svg)
![OpenCV](https://img.shields.io/badge/Image-OpenCV-green.svg)

这是一个基于 Python 开发的专业自动白平衡 (AWB) 校正工具，提供现代化的 GUI 界面，旨在修复由于环境光导致的图像色彩偏差。

## 🌟 核心特性

- **现代 UI 设计**：基于 `customtkinter` 的暗色模式圆角设计，极致的视觉体验。
- **多种算法支持**：
  - **灰度世界算法** (Gray World)：基础版与优化平均值版。
  - **完美反射体算法** (Perfect Reflector)：支持高光像素占比调节。
  - **灰度边缘算法** (Gray Edge)：基于 Sobel 边缘检测，仅对边缘区域做灰度世界校正，校正更精准。
- **全中文支持**：支持包含中文的文件路径加载与保存。
- **实时预览**：处理前后双栏实时对比。

## 📸 界面预览

*(UI.png")*

## 🚀 快速上手

### 环境安装
```bash
pip install opencv-python numpy pillow customtkinter
```

### 运行程序
```bash
python awb_gui.py
```

## 📁 项目结构
- `awb_gui.py`: 程序主入口与 GUI 实现。
- `core/`: 核心算法模块 (`awb_algorithms.py`)。
- `archive/`: 项目历史版本与备份。

## 🛠 开发计划
- [ ] 增加更多高级 AWB 算法（如机器学习方法）。
- [ ] 支持批量处理功能。
- [ ] 导出 PDF 报告功能。

## 📄 开源协议
MIT License

### 关键说明
1. **文件结构**：确保 `core` 文件夹存在，且 `awb_algorithms.py` 放在该目录下，`awb_gui.py` 与 `core` 目录同级。
2. **依赖不变**：仍使用原有的依赖（`opencv-python`/`numpy`/`pillow`/`customtkinter`），无需额外安装。
3. **功能兼容**：新增的灰度边缘算法完全兼容原有逻辑，无边缘的图片会自动退化为普通灰度世界算法。
4. **UI 交互**：
   - 选择“灰度边缘算法”时，会显示边缘检测阈值滑块（10~100）；
   - 选择“完美反射体算法”时，显示高光占比滑块；
   - 其他算法隐藏参数面板，保持界面简洁。