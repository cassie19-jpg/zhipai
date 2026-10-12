# Third-party notices / 第三方组件声明

本项目自身代码（`index.html`、`style.css`、`sw.js`、`js/` 等）以 MIT 协议发布，见 [LICENSE](LICENSE)。
以下第三方文件**未经修改**地随项目一起分发，各自遵循其原始许可证：

| 文件 | 组件 | 版本 | 许可证 | 来源 |
| --- | --- | --- | --- | --- |
| `vendor/vision_bundle.mjs`、`vendor/wasm/*` | MediaPipe Tasks Vision (`@mediapipe/tasks-vision`) | 0.10.x（npm 1.1.0 构建） | Apache-2.0 | https://github.com/google-ai-edge/mediapipe · https://www.npmjs.com/package/@mediapipe/tasks-vision |
| `models/pose_landmarker_lite.task` | MediaPipe Pose Landmarker (lite, float16) | latest | Apache-2.0 | https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker |
| `models/efficientdet_lite0.tflite` | EfficientDet-Lite0 目标检测模型（MediaPipe 发布版, float16） | latest | Apache-2.0 | https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector |
| `vendor/mediabunny.min.mjs` | Mediabunny（视频解封装/解码/编码/封装） | 1.61.3 | MPL-2.0 | https://github.com/Vanilagy/mediabunny · https://www.npmjs.com/package/mediabunny |

许可证全文：

- Apache License 2.0：[licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)
- Mozilla Public License 2.0（Mediabunny）：[licenses/MPL-2.0-mediabunny.txt](licenses/MPL-2.0-mediabunny.txt)。
  Mediabunny 的源代码可在上面的 GitHub 仓库获取；本项目没有修改它。

模型与 wasm 文件放在仓库里自托管，是为了在中国大陆也能正常加载（不依赖 Google CDN）。
