# CCGAN-VC

Shindong Lee  

Korea University

## Overview

Conditional CycleGAN Voice Converter which conducts non-parallel many-to-many voice conversion with single generator and discriminator

**CycleGAN-VC** was successful in non-parallel voice conversion for 2 speakers. CycleGAN-VC needs **2 generators and 2 discriminators for one-to-one** voice conversion.

**Conditional CycleGAN-VC (CCGAN-VC)** utilizes speaker identity vectors for additional information and conditions generator and discriminator with them. CCGAN-VC needs **only 1 generator and 1 discriminator** for conducting **many-to-many** voice conversion.  

## Performance


## Reference

---CycleGAN-VC---
"Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks" by Takuhiro Kaneko and Hirokazu Kameoka
(paper: https://arxiv.org/abs/1711.11293)  

(Tensorflow implementation by Lei Mao: https://github.com/leimao/Voice_Converter_CycleGAN)  

---Dataset---  
VCC2016 Dataset: (It's already inside the repo)  
(http://vc-challenge.org/vcc2016/summary.html)  


## How to Run
