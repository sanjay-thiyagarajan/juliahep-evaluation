# Classifier Task - Machine Learning in Julia for Calorimeter Showers

### Model Architecture  
**Type** : Neural Network  
 
**Activation function (output layer)** : Softmax  
Softmax converts the raw output of the neural network into probabilities. Each output neuron's activation represents the probability that the input belongs to the corresponding class. These probabilities sum up to 1 across all classes, making it suitable for multi-class classification.  

**Neuron configuration** : input: 3->32 | output: 32->2 | softmax  
Since these are hyperparameters, a grid search is performed to identify the best structure. There are many such hyperparameter optimization techniques and I just went with the simplest one.  

### Loss Function  
**Cross Entropy**  

Cross Entropy loss is well-suited for multi-class classification tasks due to its probabilistic interpretation, favorable gradient properties, ability to avoid saturation, interpretability, and suitability for classification problems. These qualities make it the preferred choice for training neural networks in classification tasks.

### Optimizer  
**Adam**  

Adam optimizer adapts the learning rates for each parameter individually based on the average of past gradients and the square of past gradients. This adaptive learning rate mechanism allows Adam to converge faster and more reliably compared to traditional optimizers with fixed learning rates.

### Other Hyperparameters  
**Epochs:** 10  
**Train - Validation Split:** 80-20   
Chosen through trial  

### Training loss  
![](https://raw.githubusercontent.com/sanjay-thiyagarajan/juliahep-evaluation/main/training_loss_plot.png)  

**Accuracy:** 77.765 %
