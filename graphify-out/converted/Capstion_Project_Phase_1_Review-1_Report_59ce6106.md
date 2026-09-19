<!-- converted from Capstion_Project_Phase_1_Review-1_Report.docx -->


Performance Analysis of Vision Transformers for Multi-Task Autonomous Vehicle Perception

A PROJECT REPORT

Submitted by

Ojasvi Kumar Sahu (23BHI10120)
Srishti Jindal (23BHI10098)
Dhrubo Dutta (23BHI10137)


in partial  fulfillment  for the  award  of  the  degree
of


BACHELOR OF TECHNOLOGY
in
# COMPUTER SCIENCE AND ENGINEERING




SCHOOL OF COMPUTING SCIENCE AND ENGINEERING
VIT BHOPAL UNIVERSITY
KOTHRIKALAN, SEHORE
MADHYA PRADESH - 466114

SEPT 2026







ABSTRACT
# This project looks at how Vision Transformer models can be used for types of perception in self-driving cars. It focuses on how accurate the perception's how much data is used what kind of model is chosen what sensors are used and how much computing power is needed. The work has two parts. The first part is a study that looks at five areas: how well the tasks are done how big the data set is, what model is used what sensors are used and how efficient the system is. The second part will use the results from the part to create a system that uses Vision Transformers for multiple tasks. This system will be trained using two data sets called BDD100K and A2D2. The tasks the system will handle include finding objects dividing images into parts and understanding the road scene using the data provided. The main idea is that the system must work in light and weather conditions while making sure the results are good but also not using too much computing power. The code that is attached includes a part that helps with the phase. It looks at 8,027 images from the BDD100K data set. Finds five things about the image quality: average brightness, how much the brightness changes how sharp the image is, how many parts are too dark and how many parts are too bright. The process takes the images turns them into black and white calculates the numbers gets the information about the environment and saves all the results in a file that is easy to read. Then the data is analyzed to find out what is happening and how different situations compare. It uses checks to see if the differences are real and how big they are. The code for showing the results creates charts that show how the numbers are connected graphs that show how the brightness is spread out graphs that show how the brightness changes charts that show the environment and a single place where all the information is together. The analysis of the BDD100K data shows that the way the images are made changes a lot. The average brightness is 102.13 with a change of about 24.82. The numbers for dark and too bright show a shape that is not even. Comparing day and night shows differences in brightness how clear the images are, how sharp they are and how many parts are too dark. Comparing weather and rain shows more parts that are too bright when it is raining. These results help pick the conditions to use and balance the data for future tests. The report shows that the image quality analysis is a way to start, not a replacement for the tests with Vision Transformer models. The accuracy of the models how fast they. The results, from the second phase are not included because the code provided does not have those result







TABLE OF CONTENTS














INTRODUCTION
Autonomous vehicles need perception systems that can take raw sensor data and turn it into a picture of the driving environment. Camera-based perception is very important because color images have information about people, lanes, roads, traffic signs and the overall scene.. In real life the way things look changes a lot. The same system might see light, dark light, rain, reflections, blurry images, parts that are hidden and different kinds of scenes. A model that works well in one situation may not work well when the images look different from what it was trained on. This challenge is even bigger when the system has to do tasks at the same time because each task uses different parts of the image information.
Vision Transformers are a way to study this problem because they use tokenized image parts and self-attention instead of traditional convolution methods. The original Vision Transformer showed that a pure Transformer can work directly with image patches. Later designs like the Swin Transformer were created to improve how images are represented at scales and make the system more practical for detailed tasks. ConvNeXt is also important as a convolutional model because it uses ideas from Transformers to rethink traditional convolution methods. These changes make it possible to compare CNN models with new Transformer models rather than just looking at the Transformer results alone.
The choice of BDD100K and A2D2 is based on their strengths. BDD100K was made to be a driving dataset that covers many different tasks and environments. It has a lot of variety in weather, places and tasks. A2D2 adds to this by providing camera and LiDAR data with 3D boxes, semantic segmentation and instance segmentation. The two datasets together help study both task diversity and differences in the way data is collected. The current project also adds a way to measure image quality so that environmental changes are actually measured not just described.
The attached analysis of BDD100K shows why measuring image quality is important. In the part that was studied day images had a brightness of 104.04 while night images had 46.07. The median brightness went from 103.29 to 34.39. The median Laplacian variance dropped from 399.77 during the day to 74.40 at night. The average number of underexposed pixels went up from 1.39% to 24.65%. The comparison between rainy days showed a different pattern. The average brightness went up from 96.47 to 103.31. The median sharpness went down from 359.25 to 285.69 and the median overexposure increased from 0.52% to 2.81%. These patterns support the need, for experiments that test how well perception systems work in real-world conditions.

OBJECTIVE AND PROBLEM STATEMENT

The main issue this project is trying to solve is that there isn't one way to test how well Vision Transformers work for self-driving cars. This test should look at how the models do on different tasks how big the training data is, what kind of design the model uses what kind of sensors are involved and how fast it can work. Self-driving car models are usually tested with a score but that doesn't show if the model still works when the data changes if the score depends on how much data was used or if the model can run fast enough in real time. So this project sees performance as something that has parts instead of just one number.
The first goal is to create a way to test different tasks that happen on the road. The Task Performance experiment will look at object detection, semantic segmentation and instance segmentation. For each task the right way to measure success will be used, like mean Average Precision for detection and mean Intersection over Union for segmentation.
The second goal is to see how well models use data by training them on more samples. In the project plan this is set up with 5K, 10K 20K and 40K samples.
The third goal is to compare model designs like ResNet, ConvNeXt, ViT and Swin Transformer. These will be tested in the conditions so that the difference in performance can be clearly seen.
The fourth goal is to look at how using sensors affects the results. This will compare using cameras with using cameras and LiDAR together and the data will be synchronized. This part is important for A2D2 because it has both camera and LiDAR data that match up.
The fifth goal is to measure how fast the models run. This will look at how it takes to process each image how many images can be handled each second how much memory the GPU uses how many parameters are in the model how many calculations are done and how long it takes to train. The goal isn't to find the best performing model but to see how each model balances performance with speed.
The final goal is to use what was learned in Phase 1 to build a -task Vision Transformer using BDD100K and A2D2 data. This phase will use one setup for all tasks and separate parts for each task. The code provided now only shows analysis of image quality not actual model results. So numbers about ViT accuracy how long it takes how memory it uses or how well it works with different sensors are still to come. The main problem is broken down into questions that can be answered: how performance changes, between tasks how it grows with data how different models compare, if more sensors make a difference and what the cost of good performance is.
LITRATURE REVIEW
The BDD100K dataset was introduced by Yu et al. as a large-scale driving dataset designed for heterogeneous multitask learning. Its motivation is particularly aligned with this project because autonomous-driving perception involves tasks of different complexity and existing datasets often constrain researchers to narrower problem settings. BDD100K was constructed from 100K driving videos and provides diversity in geography, environment, and weather, together with multiple task annotations. [2] This makes it suitable as the primary RGB dataset for evaluating how a common perception framework behaves across multiple road-scene tasks.

Geyer et al. introduced A2D2 as a multimodal autonomous-driving dataset containing simultaneously recorded images and 3D point clouds, together with 3D bounding boxes, semantic segmentation, instance segmentation, and automotive-bus information. The sensor suite consists of six cameras and five LiDAR units, and the released annotations include 41,277 frames with semantic segmentation labels and 12,497 frames with 3D bounding-box annotations. [1] A2D2 is consequently valuable for the sensor-configuration dimension of the present study because it makes camera-only and multimodal investigations technically plausible within the same sensing environment.

At the model level, ResNet remains a standard convolutional baseline because residual connections enable the construction of deep networks while preserving effective optimization behaviour. ConvNeXt later revisited the design space of modern ConvNets and showed that carefully modernized convolutional architectures can remain competitive with Transformer-based models. [7], [8] These works motivate the inclusion of both a conventional residual network and a contemporary convolutional backbone in the experimental comparison. They provide a reference point against which any claimed advantage of Vision Transformers can be interpreted.

Dosovitskiy et al. established the original Vision Transformer framework in which an image is divided into fixed-size patches that are embedded as a token sequence and processed with Transformer layers. [9] The strength of this approach is its ability to model long-range relationships through self-attention, but its direct application to dense visual tasks introduces challenges associated with spatial resolution and multi-scale structure. Swin Transformer addresses these challenges through hierarchical representations and shifted local windows, producing a more scalable backbone for tasks such as detection and segmentation. [10] The project therefore compares both a conventional ViT and a hierarchical Transformer rather than assuming that all Transformer designs behave identically.

For external validation and segmentation-oriented comparison, the project may draw on Cityscapes, which was designed for semantic urban-scene understanding and contains imagery from 50 cities with dense annotations for pixel-level and instance-level labelling. [5] The present report does not claim that such cross-dataset experiments have already been performed; it identifies them as a suitable external benchmark once the primary Phase 1 models are operational. Overall, the literature indicates a progression from specialized convolutional perception models toward Transformer-based unified representations, while the proposed study focuses on a controlled empirical question that cuts across architecture, task, data scale, sensing, and computation.





















PROPOSED WORK
The proposed system is set up as a two‑phase pipeline. Phase 1 starts with preparing data from BDD100K and A2D2 then it uses five evaluation modules that each change only one main factor: the task, the amount of data the model design, the sensor setup or the computing measurement. The goal is to keep preprocessing and evaluation rules the same while altering one main factor each time. This way changes in performance can be understood clearly. The full Phase 1 workflow is shown in a master dashboard and other experimental drawings that were supplied. Now the evidence only backs the data‑characterization part of the pipeline. The parts that compare models are still. Will only be added once the training and evaluation code is ready.


Fig. 1 Illustrative view of the proposed work ( Phase A ).

The first stage is preparing the dataset. BDD100K gives RGB images of road scenes and annotations for tasks. A2D2 gives camera and LiDAR data, plus extra segmentation and 3D annotations that match.

The preprocessing stage aims to make samples that're in sync and reproducible. It also normalizes the input size cleans the data and splits it into training, validation and test parts. Special care is needed to keep variety and to stop information from leaking between related frames or sequences. The image‑quality scripts already set a data‑processing pattern: images are read from directories that match each split environmental data is taken from the matching annotation JSON files and a structured record is made for every image that gets processed successfully.

Inside Phase 1 the Task Performance module gives a benchmark for each task. The Dataset Scale module creates learning curves that show how performance changes with data size. The Model Architecture module compares ResNet, ConvNeXt, ViT and Swin Transformer. The Sensor Configuration module compares setups that use camera or that use both camera and LiDAR where it is possible. The Computational Efficiency module notes the resource cost for each model.

All experiments come together at a performance‑analysis stage that looks at accuracy, efficiency, robustness and generalization. The supplied visualization uses a structure with five modules instead of a normal mind map. This layout is good for showing the logic in the final report and presentation.















Fig.2 Illustrative view of the proposed work ( Phase B ).

Phase 2 takes the evidence from Phase 1 to build the multi‑task perception model. The proposed architecture has a shared Vision Transformer encoder. Then separate prediction heads for each task. In the RGB pathway images from BDD100K and A2D2 are turned into patch. Token representations before they go through Transformer blocks. When multimodal fusion is used LiDAR features from A2D2 can be matched with the camera representation using a fusion method. The shared encoder is meant to learn scene representations that can be used again. Separate task heads then make outputs, for detection, semantic segmentation, instance segmentation and other finished tasks. The attached Phase 2 diagram is an architectural proposal. It is not a record of training results..
























HARDWARE AND SOFTWARE REQUIERMENTS
The data needs are mainly set by the two sources chosen. BDD100K is the RGB driving data set because it is made to support different tasks at the same time while A2D2 is the extra data set for seeing with more than one sense and looking at how sensors work. [1] [2] The BDD100K analysis that is attached works with 8,027 images that were properly checked out from the 10K images in the project. Each image is connected to information about the split and details about the environment like weather time of day and the kind of place. The code that checks image quality goes through the train, validation and test folders. Makes a CSV file that has numbers about light how bright the image is, percentiles, environment labels and how the metadata matches. The statistics process then makes summaries for all, day and night and clear and rainy weather.
The software setup that can be seen in the code is Python with NumPy, pandas, OpenCV, SciPy, Matplotlib handling JSON working with files and the concurrent.futures module in Python. OpenCV is used to get the images and change them to black and white. NumPy does the math with arrays and the percentiles. Pandas helps with tables. Making CSV files. SciPy does the tests and the calculations for density. Matplotlib makes the pictures that are shown. The image-quality code also uses a ThreadPoolExecutor that has a number of workers that can't go over 12 or the number of CPUs. This lets each image be handled on its own at the time. The code that makes visuals saves quality JPG and SVG pictures at 300 dots per inch for showing and reporting.
The computer needs for the image-quality analysis that is done are not very high because the process uses the CPU and works on each image to get the stats. The project doesn't say what the minimum or best CPU, RAM, storage or GPU is for the part where the Vision Transformer is trained. So those numbers should not be put as needed now. For the part of the model experiments and the second part where the Transformer is trained a GPU that works with CUDA, enough memory enough RAM and fast storage is usually needed.. The exact computer needs depend on the image size, the type of model how many images at once the way to change the model and if LiDAR is used. These numbers should be added after the time the training is done and the real needs are known.
To make sure the work can be done again the software setup should be fixed once the model starts being made. This includes the Python version, the PyTorch version, the CUDA version when needed, the computer vision tools and the random numbers. The code that is there already shows how to make the files organized and the results in a way. This includes CSV files for all the numbers and the environment comparisons and a JSON file, for the tests. The final version should do the same with the model checkpoints, the settings, the logs, the results and the tests so that each experiment can be made again from a setup instead of depending on unexplained settings.





















HARDWARE AND SOFTWARE REQUIERMENTS
1) Dataset and Metadata Preparation. Its role is to collect images and associated labels, resolve environmental attributes, preserve train/validation/test membership, and provide a consistent representation for downstream analysis. In the supplied image-quality implementation, metadata are resolved first by exact image-name matching and, when that is unavailable, by a video-prefix fallback. Weather, time of day, and scene attributes are then stored alongside image identifiers and split information. This module establishes the context required for later robustness analysis because the image-quality measurements can be stratified by environmental condition rather than treated as an undifferentiated aggregate.
2) Image-Quality Characterization. It computes mean luminance as the average grayscale intensity, luminance standard deviation as a contrast measure, Laplacian variance as the sharpness or blur indicator, and the proportions of pixels below or above predefined exposure thresholds. Additional severe-clipping indicators and luminance percentiles are also calculated by the extractor. The implemented pipeline uses the ITU-R BT.601 grayscale conversion exposed through OpenCV, calculates the metrics per image, and stores the values with controlled rounding. This module produced the 8,027-image dataset characterization used in the attached report and figures.
3) Statistical and Distributional Analysis. It computes means, standard deviations, medians, interquartile ranges, fifth and ninety-fifth percentiles, extrema, and skewness for each metric. For environmental comparisons, the implementation applies two-sided Mann–Whitney U tests, two-sample Kolmogorov–Smirnov tests, and Cohen’s d effect sizes. The resulting day-versus-night analysis shows very large differences for brightness, contrast, and underexposure, while the clear-versus-rain comparison exhibits a particularly strong difference in overexposure. The associated visualization module expresses these results through histograms with KDE curves, CDFs, correlation plots, and benchmark bar charts, allowing both distributional structure and aggregate differences to be inspected.
4) Phase 1 Model Evaluation Framework. This module is planned to train and test ResNet, ConvNeXt, ViT, and Swin Transformer backbones over the selected perception tasks under controlled protocols. Its output is intended to support the Task Performance and Model Architecture experiments and to supply comparable accuracy measures for the later computational-efficiency analysis.

5)Dataset-Scale Evaluation, which repeats the selected model training over the planned 5K, 10K, 20K, and 40K sample regimes. The resulting learning curves are intended to quantify how additional training data influence the performance of each architecture.
6)  Sensor Configuration and Multimodal Fusion. Its purpose is to compare camera-only inference with camera-plus-LiDAR inference using A2D2 data when a validated fusion implementation is available.
7) Computational Efficiency, which records latency, FPS, GPU memory, parameter count, FLOPs, and training time alongside accuracy.
8) Phase 2 Multi-Task Implementation. It is planned as a shared Vision Transformer encoder with task-specific heads for the finalized perception tasks, trained using the combined BDD100K and A2D2 datasets. The attached materials establish the data-analysis foundation and the proposed architecture, but the model-training and deployment outputs of these later modules are not yet represented by completed numerical results.

# Figures and Analytical Outputs

Fig. 3 Spearman correlation matrix of BDD100K image-quality properties


Figure 4. Environmental distributions of brightness, sharpness, underexposure, and overexposure.


Figure 4. Cumulative distribution of mean luminance across environmental domains.

Figure 5. Environmental-domain benchmarks across visual quality metrics.




Figure 6. Integrated dashboard summarizing the BDD100K image-quality analysis.



# Selected Quantitative Findings from the Attached BDD100K Analysis

For the 8,027 analysed images, the overall mean luminance is 102.13 ± 24.82, with a median of 102.62 and a 5th–95th percentile range of 63.53–140.29. Contrast has a mean of 61.32 ± 10.22 and median 61.90. Laplacian variance has a mean of 362.61 ± 269.78 and median 302.24, indicating a substantially dispersed sharpness distribution. Underexposed pixels account for 2.74% ± 7.92% on average, with a skewness of +6.13, while overexposed pixels average 1.92% ± 2.99% with skewness +7.83. The large positive skew values indicate that the exposure measures contain substantial high-end tails rather than approximately symmetric distributions.

In the day-versus-night comparison, daytime mean luminance is 104.035 and night mean luminance is 46.072, with Cohen’s d = 3.1839 and a two-sided Mann–Whitney p-value of 1.06945 × 10⁻⁸⁸. Contrast changes from 61.634 to 38.345 with Cohen’s d = 2.9296 and p = 1.95136 × 10⁻⁷⁷. Mean Laplacian variance changes from 440.541 to 169.788, with median values of 399.768 and 74.400 respectively; the effect size is d = 0.9721 and p = 1.43315 × 10⁻⁵⁸. Underexposure changes from 1.393% to 24.654%, with Cohen’s d = −4.1964 and p = 8.38979 × 10⁻⁶³. The corresponding Kolmogorov–Smirnov statistics are 0.7774, 0.7251, 0.6180, and 0.6426 for brightness, contrast, sharpness, and underexposure, respectively.

In the clear-versus-rain comparison, mean luminance changes from 96.473 to 103.308, while contrast changes from 58.013 to 62.049. Mean Laplacian variance changes from 390.434 to 362.814 and median variance from 359.251 to 285.689. The most pronounced shift occurs for overexposed pixels: the clear-weather mean is 1.203% compared with 3.205% during rain, the median rises from 0.524% to 2.813%, Cohen’s d is −0.6917, and the Mann–Whitney p-value is 9.67615 × 10⁻⁵⁵. These numerical differences support the use of environmental condition as an explicit stratification factor in subsequent robustness experiments.




REFERENCES
Geyer, J., Kassahun, Y., Mahmudi, M., Ricou, X., Durgesh, R., Chung, A.S., Hauswald, L., Pham, V.H., Mühlegg, M., Dorn, S., Fernandez, T., Jänicke, M., Mirashi, S., Savani, C., Sturm, M., Vorobiov, O., Oelker, M., Garreis, S. and Schuberth, P., A2D2: Audi Autonomous Driving Dataset, arXiv preprint arXiv:2004.06320, 2020.
Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V. and Darrell, T., BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning, arXiv preprint arXiv:1805.04687, 2020.
Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G. and Beijbom, O., nuScenes: A Multimodal Dataset for Autonomous Driving, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, 11621–11631.
Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Patnaik, V., Tsui, P., Guo, J., Zhou, Y., Chai, Y., Caine, B., Vasudevan, V., Han, W., Ngiam, J., Zhao, H., Timofeev, N., Ettinger, S., Krivokon, M., Gao, A., Joshi, A., Zhang, Y., Shlens, J., Chen, Z. and Anguelov, D., Scalability in Perception for Autonomous Driving: Waymo Open Dataset, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, 2446–2454.
Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S. and Schiele, B., The Cityscapes Dataset for Semantic Urban Scene Understanding, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, 3213–3223.
Geiger, A., Lenz, P., Stiller, C. and Urtasun, R., Vision Meets Robotics: The KITTI Dataset, The International Journal of Robotics Research, 2013.
He, K., Zhang, X., Ren, S. and Sun, J., Deep Residual Learning for Image Recognition, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, 770–778.
Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T. and S. Xie, A ConvNet for the 2020s, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, 11976–11986.
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. and Houlsby, N., An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale, International Conference on Learning Representations (ICLR), 2021.
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. and Guo, B., Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows, Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, 10012–10022.
| CHAPTER NO. | TITLE | PAGE NO. |
| --- | --- | --- |
|  | List of Abbreviations		
List of Figures and Graphs			
List of Tables		
Abstract | iii
iv
v
vi |
| 1 | PROJECT DESCRIPTION AND OUTLINE 
Introduction		
1.2 	Motivation for the work          	
1.3	[About Introduction to the project     
            including techniques]
1.5 	Problem Statement		
1.6 	Objective of the work		
1.7 	Organization of the project
           1.8	Summary | 1

.
.
. |
| 2 | RELATED WORK INVESTIGATION 
2.1 	Literature 			
2.2 	Existing Approaches/Methods 			
             2.3.1	  Approaches/Methods -1			
             2.3.2	  Approaches/Methods -2
2.3.3	  Approaches/Methods -3 	
2.4 	<Pros and cons of the stated  Approaches/Methods >
2.5 	Issues/observations from investigation   
2.6	Summary |  |
| 3 | REQUIREMENT ARTIFACTS
3.1	Introduction
3.2	Hardware and Software requirements
            3.3	Specific Project requirements
            3.3.1 Data requirement
            3.3.2 Functions requirement
            3.3.3 Performance and security requirement
           3.3.4  Look and Feel Requirements
	3.4	Summary |  |
| 4 | CHAPTER-4:
DESIGN METHODOLOGY AND ITS NOVELTY
            4.1     Methodology and goal 
4.2      Functional modules design and analysis
4.3     Software Architectural designs
4.4     Subsystem services
4.5     User Interface designs
4.5     ………………..
            4.6     Summary |  |
| 5 | CHAPTER-5:
TECHNICAL IMPLEMENTATION & ANALYSIS
           5.1    Outline
           5.2    Technical coding and code solutions
           5.3    Working Layout of Forms
           5.4    Prototype submission
           5.5    Test and validation
           5.6    Performance Analysis(Graphs/Charts)
           5.7     Summary |  |
| 6 | CHAPTER-6:
PROJECT OUTCOME AND APPLICABILITY
         6.1    Outline
         6.2    key implementations outlines of the System
         6.3    Significant project outcomes
         6.4    Project applicability on Real-world applications
         6.4    Inference |  |
| 7 | CHAPTER-7:
CONCLUSIONS AND RECOMMENDATION
         7.1    Outline
         7.2    Limitation/Constraints of the System
         7.3    Future Enhancements
        7.4    Inference |  |
|  | Appendix A
          Appendix B
          References
Note: List of References should be written as per IEEE/Springer reference format.   (Specimen attached) |  |