# EECS 545-ML-Project

Training pipeline is illustrated below: 

 In our novel proposed method, we input an unlabelled image and pre-defined prompt to the pretrained VirTex-v2 model to generate captions along with logits corresponding to each token in the caption. Using a noun extraction module, we will extract all the nouns from the caption. Now, using the nouns and their corresponding logits we pass it through a target extraction module to generate our pseudo class labels by comparing similarity between each word and the class labels of our downstream dataset. Using these pseudo class labels, we pass it through a class activation method namely GradCAM to generate heatmaps. Using these heatmaps, we draw bounding boxes on the image and obtain their respective coordinates. We also perform a post-processing step to discard redundant bounding boxes with the same class label according to the logit score threshold.
 
 Finally, we train a FCOS object detection model with our best performing model.

 ![image](https://user-images.githubusercontent.com/73309689/181842456-13ced9ac-4fd3-4e27-9c5d-fc4b410a4462.png)
