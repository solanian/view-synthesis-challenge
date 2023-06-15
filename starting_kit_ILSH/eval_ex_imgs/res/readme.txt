Runtime per image[s]: 10.43
CPU[0] / GPU[1]: 1
No Extra Data [0] / Extra Data [1]: 0
Data-part among validation [0] or test [1]: 0
Other description: The solution uses the MethodA of Jang et al. ICCV 2022 as a base model. On top of the base model, we applied a novel ray selection method, which helps the proposed neural rendering model initiate the sampling process efficiently and reliably. We have a Python/C++ implementation and report single-core CPU runtime. The method was trained mainly on the Imperial Light-Stage Head dataset, but in the middle of training, we added additional head images from DatasetB, which were introduced in CVPR 2022 by Young et al. to specifically outperform the case of artifactC.