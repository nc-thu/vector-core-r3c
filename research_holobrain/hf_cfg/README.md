---
license: apache-2.0
---

<div align="center">
  <img src="https://github.com/HorizonRobotics/robot_lab/blob/master/holobrain/assets/holobrain_logo.png?raw=true" alt="HoloBrain Logo" width="400" style="vertical-align: middle; margin-right: 15px;">
  <h1 style="display: inline-block; margin: 10; font-size: 2em">A foundation model for general embodied manipulation</h1>
</div>

<div align="center" class="authors">
Xuewu Lin, Yun Du, Hongyu Xie, Yiwei Jin, Jiawei Li, Shijie Wu, Qingze Wang, Mengao Zhao, Ziang Li, Chaodong Huang, Mengdi Li, Hongzhe Bi, Lichao Huang, Zhizhong Su, Tianwei Lin
</div>

<div align="center" style="line-height: 3;">
  <a href="https://horizonrobotics.github.io/robot_lab/holobrain/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/🏠HoloBrain-HomePage-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2602.12062" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/📄Paper-arXiv-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/HorizonRobotics/RoboOrchardLab/tree/master/projects/holobrain/" target="_blank" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/💻Code-Github-black" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/collections/HorizonRobotics/holobrain" target="_blank" style="margin: 2px;">
    <img alt="Model" src="https://img.shields.io/badge/⚙️HoloBrain Model-HuggingFace-orange" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


## 📘 Framework
<div align="center">
  <img src="https://github.com/HorizonRobotics/robot_lab/blob/master/holobrain/assets/holobrain_framework.png?raw=true" width="90%" alt="HoloBrain" />
  <p style="font-size:1em; color:#555;">By incorporating explicit embodiment modeling (e.g., camera parameters and kinematic descriptions), our model effectively unifies training across heterogeneous robots. Together with a full-stack VLA infrastructure (RoboOrchard) and an effective test-driven data strategy, HoloBrain-0 delivers superior performance on both real world and simulation manipulation benchmarks.</p>
</div>


## 📁 Quick Start
The exported model and processor can be used very conveniently. You can insert the code below into any location to perform model inference.
```python
from robo_orchard_lab.models.holobrain.processor import (
  HoloBrainProcessor,
  MultiArmManipulationInput,
  MultiArmManipulationOutput,
)
from robo_orchard_lab.models.mixin import ModelMixin

# load model and processor
processor = HoloBrainProcessor.load("./HoloBrain_v0.0_GD", "robotwin2_0_processor.json")
model = ModelMixin.load_model("./HoloBrain_v0.0_GD/pretrain", load_impl="native")

input_data: MultiArmManipulationInput
input_data = processor.pre_process(input_data)
model_outs = model(input_data)
output_data: MultiArmManipulationOutput = processor.post_process(input_data, model_outs)
```

## 📄 Citation
```
@misc{lin2026holobrain0technicalreport,
      title={HoloBrain-0 Technical Report},
      author={Xuewu Lin and Tianwei Lin and Yun Du and Hongyu Xie and Yiwei Jin and Jiawei Li and Shijie Wu and Qingze Wang and Mengdi Li and Mengao Zhao and Ziang Li and Chaodong Huang and Hongzhe Bi and Lichao Huang and Zhizhong Su},
      year={2026},
      eprint={2602.12062},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.12062},
}
```
