
Neural Font Classification for Ancient Text Digitization
- Barontini Chiara, Milanino Tommaso, Sandu Andrei
1. Introduction – Our project classifies digitalized pages of ancient text based on their font. Part of our Machine Learning coursework, we developed a model to assist the "Librarian of Alexandria" in organizing the documents based on the fonts present within them. More than 1000 scanned pages made up the dataset for this project. Given the different origins of the documents, with them being from various years and containing different writing styles, a Convolutional Neural Network has been the main focus of this project. 

    - Our work was compiled of: 
        - The initial exploratory data analysis
        - Image Preprocessing
        - Data Augmentation
        - Building the Convolutional Neural Network

2. Methods: This section presents the techniques used on our dataset for exploration, preprocessing, data augmentation, and model development.

   - 2.1 Exploratory analysis and preprocessing: The dataset used for this project consists of 1,256 ancient digital texts labelled based on 11 distinct font classes: augustus, aureus, cicero, colosseum, consul, forum, laurel, roman, senatus, trajan, and vesta. An initial analysis of the distribution of samples across font categories revealed that overall, the classes were well represented with slight imbalances, for instance, cicero and aureus appeared more frequently than forum and laurel. Next, we inspected the dimensions of the images in the dataset which revealed a lot of variability in the width and height of the images. We also checked for duplicate or corrupt images and found one. Given the variability in the images, some standard preprocessing steps were applied to ensure consistency. First, grayscale conversion was applied to reduce complexity, then to each image was applied Gaussian blurring using a 3x3 kernel to eliminate noise, helpful to ignore minor imperfections often present in scanned documents. Following denoising, we applied Otsu’s thresholding to convert each image into a binary black and white format, this improved the contrast between text and background making fonts more distinguishable to the model. Finally, images were resized to a standard dimension of 224x224 to normalize input size across the dataset and reduce computation cost.

   - 2.2 Data Augmentation: Given the limited size of the dataset, data augmentation was an important method to improve robustness and generalization of our models. The transformations simulate possible distortion that could happen in real life, making the model more robust to small variations. First, we split the dataset into train and test set, then we applied the augmentation only on the training set while leaving the test set untouched, since we want to evaluate on original images from the dataset. Augmentation was implemented using ImageDataGenerator from Keras, which applies augmentation in real-time while training. The augmentations include applying random rotations, horizontal and vertical shifts, zoom, horizontal vertical flips, shearing, brightness change. For the first model (explained in the next section) we applied more aggressive data augmentation, while for the second model we used a more conservative augmentation to preserve the integrity of pretrained weights.

   - 2.3 Model Architectures: To explore different deep learning methods, two distinct neural network architectures were used. The baseline CNN was designed with a hierarchical structure of four convolutional blocks, each made up of two convolutional layers, to extract spatial features, combined with batch normalization, for training stability, followed by a maxpooling layer, for downsampling. The model began with 32 filters and progressively increased to 256, allowing the network to learn more complex features. A fully connected layer followed by a dropout and a final softmax output layer was used for classification. The model was trained using Adam optimizer with low learning rate, categorical crossentropy as the loss function, since our task is multi-class classification, and accuracy as the evaluation metric. The second model employed a CNN exploiting the EfficientNet architecture pre-trained on ImageNet. Since EfficientNet expects rgb inputs, grayscale images were converted to three channels. Then, 80% of the base layers of EfficientNet were partially frozen to retain learned features while allowing the upper layers to fine tune and adapt to our domain-specific font classification task. A global average pooling was used to reduce the spatial dimansions, followed by a dense layer, Relu, batch normalization, and a final softmax layer. This architecture benefited from the pre-learned features, allowing faster convergence and improved accuracy.

   - 2.4 Environment: Our project was developed and tested on Google Colab, using Python 3.11.12 on T4 GPU. The deep learning models were implemented in TensorFlow 2.18.0 along with supporting libraries such as Keras, Pandas, Matplotlib, Seaborn, and Scikit-learn (tensorflow=2.18.0, numpy=2.0.2, pandas=2.2.2, matplotlib=3.10.0, seaborn=0.13.2, scikit-learn=1.6.1).

4. Experimental Design: To approach the problem of font classification, we experimented with two deep learning models (described in detail in the previous section), a custom CNN designed from scratch without external knowledge and a transfer learning model with EfficientNet architecture pre-trained on ImageNet. The first serves as a baseline to understand how well models created from scratch perform on this task, while the second model provides a state-of-the-art benchmark to understand the benefit of using pre-trained models in our specific case. Next, we aimed to evaluate whether using the first or the second architecture yields better results, given our limited dataset. To evaluate model performance, we used the following evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix, training and validation curves. Accuracy was used to measure how often the model makes correct predictions in general, while precision recall, and F1-score were used to understand the performance per font to see if there was any imbalance. The confusion matrix visualizes predictions versus actual labels across all fonts, this is useful for identifying frequent misclassifications. We also used training and validation curves to visualize accuracy and loss across epochs during training, this helped us understand if accuracy and loss were stable or fluctuated a lot during training and to see if the model reached a plateau after a certain epoch.

5. Results –  Our two convolutional neural networks used a classic 80/20 train test split to ensure consistency and comparability. To monitor overfitting, we implemened callbacks such as EarlyStopping and ReduceLRonPlateau, assuring the validity of our results. Both of our CNNs used real-time data augmentation during training, including random rotations, zoom, shifts and flipping which were applied to the resized images. Each model used a distinct set of augmentation parameters. This was done to achieve our aim of investigating the model's response to previously unseen historical text images. Model 1 presented promising statistics. However, the accuracy was not up to par with that required in practical use in the context of archival or digitization workflows where reliability is essential. The results of Model 1 can be seen below: 

    ![Model 1 Accuracy](readme_accomp/Model_1_Accuracy.png)

    *Figure 1: Accuracy statistics for Model 1 during training and validation*

    In terms of model accuracy, both training and validation accuracy increase steadily, reaching around 62% by epoch 50. The plot of the validation line, which fluctuates early on and stabilizes as we reach epoch 35, suggests that the model is learning generazible features rather than overfitting. 

    ![Model 1 Loss](readme_accomp/Model_1_Loss.png)

    *Figure 2: Loss statistics for Model 1 during training and validation*

    Training and validation losses decrease consistently. Validation loss initially spikes which is likely a result of the challenging nature of the project. Later, the validation follows the training loss, again suggesting a lack of overfitting. 

    Model 2 performed significantly better than Model 1. Model 2 presented results which emphasized its usability in the context of archiving, making it appropriate for use in the case of the Library of Alexandria while boasting a 96% accuracy score. The detailed results for Model 2 are presented below:

    ![Model 2 Accuracy](readme_accomp/Model_2_Accuracy.png)

    *Figure 3: Accuracy statistics for Model 2 during training and validation.*

    Both training and validation accuracy improve rapidly in the first few epochs, with the training accuracy breaking the 90% mark by epoch 5. Past the epoch 5 mark, both curves continue to rise slowly, with training accuracy presenting a value of 96% at epoch 30 and validation accuracy presenting a slightly lower accuracy score. 

    ![Model 2 Loss](readme_accomp/Model_2_Loss.png)

    *Figure 4: Loss statistics for Model 2 during training and validation*

    Model 2's training loss consistently decreases over time, starting at a threshold that Model 1 never managed to pass. Training loss approaches near-zero values by epoch 30, with validation loss following a similar trend but presenting a larger final value of approximately 0.25. This indicates good generalization. The gap between the training and validation loss curves once again confirms the model's avoidance of overfitting, learning from the data rather than memorizing specific examples from the dataset. 


    ### Classification Report for Model 2

    | Font Class | Precision | Recall | F1-Score | Support |
    |------------|-----------|--------|----------|---------|
    | augustus   | 0.91      | 0.91   | 0.91     | 22      |
    | aureus     | 0.97      | 1.00   | 0.98     | 29      |
    | cicero     | 0.96      | 1.00   | 0.98     | 27      |
    | colosseum  | 0.96      | 1.00   | 0.98     | 23      |
    | consul     | 1.00      | 1.00   | 1.00     | 18      |
    | forum      | 1.00      | 1.00   | 1.00     | 17      |
    | laurel     | 1.00      | 0.94   | 0.97     | 18      |
    | roman      | 0.96      | 0.88   | 0.92     | 26      |
    | senatus    | 0.96      | 0.92   | 0.94     | 24      |
    | trajan     | 1.00      | 1.00   | 1.00     | 20      |
    | vesta      | 0.92      | 0.96   | 0.94     | 24      |

    | **Metric**       | **Score** |
    |------------------|-----------|
    | Accuracy         | 0.96      |
    | Macro Avg F1     | 0.97      |
    | Weighted Avg F1  | 0.96      |

    The classification report showcases the strong performance across all 11 font classes, with an overall accuracy of 96%. Most of the fonts showcase an F1-score of over 0.95 with the exception of the augustus font which the model had a harder time in classifying when compared to the rest. Fonts like consul, forum and trajan were classified with 100% accuracy. All of these results together highlight the excellent performance of the model in recognizing and classifying the fonts from the sample. 

    ### Top 10 Worst Mistakes

    - **True:** *roman* → **Predicted:** *augustus* (Mistakes: 2)  
    - **True:** *augustus* → **Predicted:** *vesta* (Mistakes: 2)  
    - **True:** *senatus* → **Predicted:** *cicero* (Mistakes: 1)  
    - **True:** *senatus* → **Predicted:** *roman* (Mistakes: 1)  
    - **True:** *roman* → **Predicted:** *senatus* (Mistakes: 1)  
    - **True:** *vesta* → **Predicted:** *colosseum* (Mistakes: 1)  
    - **True:** *laurel* → **Predicted:** *aureus* (Mistakes: 1)  

    The most common mistakes made by the model are presented above. Out of these, the mistakes which were most present were the model mistaking the roman font for the august font and the augustus font being mistakem for the vesta font, both of these mistakes were made twice by the model. These mistakes are also visualised in the correlation matrix presented below. 

    ![Model 2 Confusion Matrix](readme_accomp/Model_2_Correlation_Matrix.png)
    *Figure 5: Confusion Matrix for Model 2*

6. Conclusions – List some concluding remarks. In particular: o Summarize in one paragraph the take-away point from your work. o Include one paragraph to explain what questions may not be fully answered by your work as well as natural next steps for this direction of future work.

