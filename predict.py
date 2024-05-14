import torch 
import torch.nn.functional as F 
from model import guide, model 
import numpy as np 
from matplotlib import colors
import matplotlib.pyplot as plt


NUMBER_OF_SAMPLES = 100
CLASSES = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9') 
 
def predict_helper(x, number_of_samples = NUMBER_OF_SAMPLES):
    ''' 
    Generates predictions by sampling models from the posterior 
    using the guide. Computes the mean prediction over all samples.

    Parameters:
        x (torch.Tensor): The input features, flattened if necessary.
        number_of_samples (int): The number of model 
        samples to draw for making predictions.

    Returns:
        torch.Tensor: The predicted class indices based 
        on the average prediction from the sampled models.
    '''
    sampled_models = [guide(None, None) for sample_index in range(number_of_samples)]
    predicted = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(predicted), 0)
    return torch.argmax(mean, axis=1)

def predict(data_loader, number_of_samples = NUMBER_OF_SAMPLES): 
    ''' 
    Evaluates the Bayesian neural network model on a 
    dataset by calculating accuracy, precision, recall, and F1 score.

    Parameters:
        data_loader (DataLoader): The DataLoader providing the dataset for evaluation.
        number_of_samples (int): The number of model samples to draw for predictions.

    Returns:
        dict: A dictionary containing total images processed, 
        prediction accuracy, precision, recall, and F1 score.
    '''
    correct = 0
    total = 0
    num_classes = 10 
    true_positive = 0
    false_positive = 0
    false_negative =0
    for j, data in enumerate(data_loader):
        images, labels = data
        predicted = predict_helper(images.reshape(-1,28*28), number_of_samples)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() 
        for i in range(num_classes):
            true_positive +=  ((predicted == i) & (labels == i)).sum().item() 
            false_positive += ((predicted == i) & (labels != i)).sum().item()
            false_negative  += ((predicted != i) & (labels == i)).sum().item()
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("accuracy: %d %%" % (100 * correct / total)) 
    return {"Total images": total, 
            "Prediction Accuracy": (100 * correct / total), 
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
    }



####################################################################################
#################################################################################### 
#################################################################################### 

##  bayesian approach: 
def give_uncertainities(x, number_of_samples = NUMBER_OF_SAMPLES):
    ''' 
    Computes the softmax probabilities for each class 
    over multiple model samples to estimate uncertainty.

    Parameters:
        x (torch.Tensor): Input tensor of images, typically preprocessed.
        number_of_samples (int): The number of sampled models to generate predictions.

    Returns:
        numpy.ndarray: Array of log softmax probabilities from each sampled model.
    '''
    sampled_models = [guide(None, None) for _ in range(number_of_samples)]
    yhats = [F.log_softmax(model(x.view(-1,28*28)).data, 1).detach().numpy() for model in sampled_models]
    return np.asarray(yhats)


def imshow(img):
    ''' 
    Displays an image by unnormalizing and using matplotlib to plot.

    Parameters:
        img (torch.Tensor): The image tensor to display.
    '''
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(npimg,  cmap='gray', interpolation='nearest')
    plt.show()

def test_batch(images, labels, number_of_samples=NUMBER_OF_SAMPLES, plot=True):
    ''' 
    Tests a batch of images and labels to calculate and plot 
    model predictions with uncertainty, evaluating metrics such as precision and recall.

    Parameters:
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): Corresponding labels for the images.
        number_of_samples (int): Number of model samples used for uncertainty estimation.
        plot (bool): Whether to plot histograms and images.

    Returns:
        tuple: Metrics including total number of images, 
        number of correct predictions, number predicted, precision, recall, and F1 score.
    '''
    y = give_uncertainities(images, number_of_samples)
    predicted_for_images = 0
    correct_predictions = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(labels)):
        real_label = labels[i].item()
        if plot:
            print("Real: ", real_label)
            fig, axs = plt.subplots(1, 10, sharey=True, figsize=(20, 2))

        all_digits_prob = []
        highlighted_something = False
        for j in range(len(CLASSES)):
            histo_exp = [np.exp(y[z][i][j]) for z in range(y.shape[0])]
            prob = np.percentile(histo_exp, 50)
            all_digits_prob.append(prob)

            if prob > 0.2:
                highlighted_something = True
                if plot:
                    N, bins, patches = axs[j].hist(histo_exp, bins=8, color="lightgray", lw=0,
                                                   weights=np.ones(len(histo_exp)) / len(histo_exp), density=False)
                    axs[j].set_title(f"{j} ({round(prob, 2)})")
                    fracs = N / N.max()
                    norm = colors.Normalize(fracs.min(), fracs.max())
                    for thisfrac, thispatch in zip(fracs, patches):
                        thispatch.set_facecolor(plt.cm.viridis(norm(thisfrac)))

        if plot and highlighted_something:
            plt.show()

        predicted = np.argmax(all_digits_prob)
        if highlighted_something:
            predicted_for_images += 1
            if real_label == predicted:
                correct_predictions += 1
                true_positive += 1
            else:
                false_negative += 1
            if predicted != real_label:
                false_positive += 1

        if plot:
            imshow(images[i].squeeze())

    # Calculate metrics
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    if plot:
        print("Summary")
        print("Total images: ", len(labels))
        print("Predicted for: ", predicted_for_images)
        print("Accuracy when predicted: ", correct_predictions / predicted_for_images if predicted_for_images else 0)
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    return len(labels), correct_predictions, predicted_for_images, precision, recall, f1_score

def bayeisan_predict(data_loader, number_of_samples = NUMBER_OF_SAMPLES): 
    ''' 
    Calculates precision, recall, and F1 score based on true positives, false positives, and false negatives.

    Parameters:
        tp (int): True positives count.
        fp (int): False positives count.
        fn (int): False negatives count.
        predicted_for (int): Number of images with predictions.
        correct (int): Number of correct predictions.
        total (int): Total number of images.

    Returns:
        dict: Dictionary containing precision, recall, F1 score, and other related metrics.
    '''
    correct = 0
    total = 0
    total_predicted_for = 0
    for j, data in enumerate(data_loader):
        images, labels = data
        total_minibatch, correct_minibatch, predictions_minibatch, precision, recall, f1_score = test_batch(images, labels, number_of_samples = number_of_samples, plot=False)
        total += total_minibatch
        correct += correct_minibatch
        total_predicted_for += predictions_minibatch
    return {"Total images": total, 
            "Skipped": total-total_predicted_for, 
            "Prediction Accuracy": (100 * correct / total_predicted_for), 
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
    }