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
    sampled_models = [guide(None, None) for sample_index in range(number_of_samples)]
    predicted = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(predicted), 0)
    return torch.argmax(mean, axis=1)

def predict(data_loader, number_of_samples = NUMBER_OF_SAMPLES): 
    correct = 0
    total = 0
    for j, data in enumerate(data_loader):
        images, labels = data
        predicted = predict_helper(images.reshape(-1,28*28), number_of_samples)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("accuracy: %d %%" % (100 * correct / total)) 
    return (100 * correct / total) 



####################################################################################
#################################################################################### 
#################################################################################### 

##  bayesian approach: 
def give_uncertainities(x, number_of_samples = NUMBER_OF_SAMPLES):
    sampled_models = [guide(None, None) for _ in range(number_of_samples)]
    yhats = [F.log_softmax(model(x.view(-1,28*28)).data, 1).detach().numpy() for model in sampled_models]
    return np.asarray(yhats)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(npimg,  cmap='gray', interpolation='nearest')
    plt.show()


def test_batch(images, labels, plot=True):
    y = give_uncertainities(images)
    predicted_for_images = 0
    correct_predictions=0

    for i in range(len(labels)):
        if(plot):
            print("Real: ",labels[i].item())
            fig, axs = plt.subplots(1, 10, sharey=True,figsize=(20,2))
        all_digits_prob = []
        highted_something = False
        for j in range(len(CLASSES)):
            highlight=False
            histo = []
            histo_exp = []
            for z in range(y.shape[0]):
                histo.append(y[z][i][j])
                histo_exp.append(np.exp(y[z][i][j]))
            prob = np.percentile(histo_exp, 50) #sampling median probability
            if(prob>0.2): #select if network thinks this sample is 20% chance of this being a label
                highlight = True #possibly an answer
            all_digits_prob.append(prob)
            if(plot):
                N, bins, patches = axs[j].hist(histo, bins=8, color = "lightgray", lw=0,  weights=np.ones(len(histo)) / len(histo), density=False)
                axs[j].set_title(str(j)+" ("+str(round(prob,2))+")") 
            if(highlight):
                highted_something = True
                if(plot):
                    # We'll color code by height, but you could use any scalar
                    fracs = N / N.max()
                    # we need to normalize the data to 0..1 for the full range of the colormap
                    norm = colors.Normalize(fracs.min(), fracs.max())
                    # Now, we'll loop through our objects and set the color of each accordingly
                    for thisfrac, thispatch in zip(fracs, patches):
                        color = plt.cm.viridis(norm(thisfrac))
                        thispatch.set_facecolor(color)
        if(plot):
            plt.show() 
        predicted = np.argmax(all_digits_prob)
        if(highted_something):
            predicted_for_images+=1
            if(labels[i].item()==predicted):
                if(plot):
                    print("Correct")
                correct_predictions +=1.0
            else:
                if(plot):
                    print("Incorrect :()")
        else:
            if(plot):
                print("Undecided.")
        if(plot):
            imshow(images[i].squeeze()) 
    if(plot):
        print("Summary")
        print("Total images: ",len(labels))
        print("Predicted for: ",predicted_for_images)
        print("Accuracy when predicted: ",correct_predictions/predicted_for_images)
    return len(labels), correct_predictions, predicted_for_images

def bayeisan_predict(data_loader): 
    correct = 0
    total = 0
    total_predicted_for = 0
    for j, data in enumerate(data_loader):
        images, labels = data
        total_minibatch, correct_minibatch, predictions_minibatch = test_batch(images, labels, plot=False)
        total += total_minibatch
        correct += correct_minibatch
        total_predicted_for += predictions_minibatch
    return {"Total images": total, 
            "Skipped": total-total_predicted_for, 
            "Prediction Accuracy": (100 * correct / total_predicted_for)
    }