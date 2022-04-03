# Flood-Mapping-with-SAR-Intensity-and-InSAR-Coherence
Improving Semantic Water Segmentation by Fusing Sentinel-1 Intensity and Interferometric Synthetic Aperture Radar (InSAR) Coherence Data


In this study, Sentinel-1 synthetic aperture radar intensity and interferometric coherence data were fused in binary classification models to improve semantic segmentation of water pixels at 10-meter spatial resolution. We considered three different model scenarios: a co-event intensity, a bi-temporal intensity, and a bi-temporal intensity and coherence model. For each scenario semantic segmentation models were trained to assess the relative improvement gained by fusing intensity and interferometry data cross-trained with Sentinel-2 derived water masks. The semantic segmentation models include an Attention U-Net model capable of accounting for the spatial distribution of the Sentinel-1 scenes and a pixel-wise XGBoost classifier. By fusing SAR intensity with interferometric coherence, we exploited the spatial decorrelation in the coherence maps during a flooding event compared to the coherence before the flooding event. The data set used in this study leverages the publicly available georeferenced Sen1Floods11 data set [1]. We augmented the co-event Sentinel-1 intensity data provided by the Sen1Floods11 team with pre-event intensity data from Google Earth Engine and InSAR products produced on-demand by the Alaska Satellite Facility [2]. Our experiment results showed that fusing the Sentinel-1 intensity data with the interferometry data improves water intersection over union results by up to 3.29% with the Attention U-Net model and up to 3.51% with the XGBoost model. 

Moreover, the results highlight that our Attention U-Net and XGBoost models systematically reduce the water-class false negative rate. Reducing the water-class false negative rate is important in flood mapping applications because this means we are not incorrectly labeling non-flooded pixels. On the other hand, the co-event and bi-temporal intensity models tend to reduce the false positive rate compared to the bi-temporal intensity and coherence models. This means that with the introduction of the coherence data, our bi-temporal intensity and coherence models tend to over-estimate water pixels more than the co-event and bi-temporal intensity models. In practice, over-estimating flooded pixels is less risky than under-estimating them. This could mean that aid may be sent to a potentially flooded area as opposed to no aid being sent at all. 

Lastly, our results also highlighted the XGBoost models' ability to outperform the Attention U-Net models with our generalization data set. Convolutional neural network models are attractive because the convolutional operations are able to account for the spatial variation in the satellite images. However, the advantages gained by the CNN models come at the expense of loss of feature interpretability. In our study, raw pizel data was used as input features to our XGBoost models. This means that we could potentially assess the relative improvements gained by adding each data modality. Moreover, the training time needed to fit an XGBoost model is orders of magnitude lower than the training time required to train a CNN model. In our experiments, training the biggest XGBoost model took approximately 3 minutes compared to about 3 hours needed to train the Attention U-Net models using the same GPU. Training time can be a differentiating factor when trying to respond quickly to extreme flooding events.

### References

[1]  D. Bonafilia, B. Tellman, T. Anderson, and E. Issenberg, “Sen1floods11: a georeferenced dataset to train and test deep learning flood algorithms for sentinel-1,”
2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 835–845, 2020.

[2] ASF, “Alaska Satellite Facility, UAF,” Accessed: Fall 2021, Spring 2022. [Online]. Available: https://asf.alaska.edu
