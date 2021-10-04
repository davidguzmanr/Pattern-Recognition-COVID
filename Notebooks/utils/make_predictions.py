import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage.transform import resize
from IPython.display import clear_output

def color_mask(img, size=512):
    # dict_labels = {'fondo':0, 'vidrio':1, 'consolidación':2, 'pleura':3}
    # A cada label le ponemos un rgba
    dict_color = {0:[0,0,0,255], 1:[55,126,184,255], 2:[77,175,74,255], 3:[152,78,163,255]}
    
    img_color = np.zeros((size,size,4))
    img = img.reshape(size,size)
    
    for i in range(size):
        for j in range(size):
            img_color[i,j] = dict_color[img[i,j]]
        
    return np.array(img_color)/255

def create_mask(pred_mask, i=0):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[i]

def resize_mask(x, size):
    return resize(x, (size,size), order=0, mode = 'constant', preserve_range = True, 
                  anti_aliasing=False, clip=True)

def show_training_predictions(model, x_val, y_val, size, epoch, logs):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(14,7))
    ax0.imshow(x_val[1]) 
    ax0.axis('off')
    ax0.set_title('Original image', size=16)
    
    ax1.imshow(color_mask(y_val[1].reshape(size,size), size=size))
    ax1.axis('off')
    ax1.set_title('Mask', size=16)
    
    pred_mask = model.predict(x_val[0:10])
    ax2.imshow(color_mask(create_mask(pred_mask, 1).numpy().reshape(size,size), size=size))
    ax2.axis('off')
    ax2.set_title('Prediction', size=16)
    
    mask_labels = np.ones((512, 32, 4))
    mask_labels[140:172] = np.array([0, 0, 0, 1])
    mask_labels[204:236] = np.array([55, 126, 184, 255])/255
    mask_labels[268:300] = np.array([77, 175, 74, 255])/255
    mask_labels[332:364] = np.array([152, 78, 163, 255])/255
    
    box = ax3.get_position()   
    box.x0 = box.x0 - 0.07
    box.x1 = box.x1 - 0.07
    ax3.set_position(box)
    ax3.imshow(mask_labels)
    ax3.tick_params(axis='x', which='both', bottom=False, top=False,labelbottom=False)
    ax3.tick_params(axis='y', which='both', left=False, labelleft=False, right = True, 
                   labelright=True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.set_yticks([156, 220, 284, 348])
    ax3.set_yticklabels(['Background','Ground glass','Consolidation','Pleural effusion'],
                        size=14)
    
    # acc, val_acc = logs['accuracy'], logs['val_accuracy']   
    plt.suptitle('Epoch {}'.format(epoch+1), size=18, y=0.8)
    fig.savefig('gif/{}_{}.jpg'.format(size, epoch), bbox_inches='tight')
    
    plt.show()

def show_predictions(img, mask, mask_prediction):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(14,7))
    ax0.imshow(img, cmap='gray') 
    ax0.axis('off')
    ax0.set_title('Original image', size=16)
    
    ax1.imshow(color_mask(mask))
    ax1.axis('off')
    ax1.set_title('Mask', size=16)
    
    ax2.imshow(color_mask(mask_prediction))
    ax2.axis('off')
    ax2.set_title('Prediction', size=16)
    
    mask_labels = np.ones((512, 32, 4))
    mask_labels[140:172] = np.array([0, 0, 0, 1])
    mask_labels[204:236] = np.array([55, 126, 184, 255])/255
    mask_labels[268:300] = np.array([77, 175, 74, 255])/255
    mask_labels[332:364] = np.array([152, 78, 163, 255])/255
    
    box = ax3.get_position()   
    box.x0 = box.x0 - 0.07
    box.x1 = box.x1 - 0.07
    ax3.set_position(box)
    ax3.imshow(mask_labels)
    ax3.tick_params(axis='x', which='both', bottom=False, top=False,labelbottom=False)
    ax3.tick_params(axis='y', which='both', left=False, labelleft=False, right = True, 
                   labelright=True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.set_yticks([156, 220, 284, 348])
    ax3.set_yticklabels(['Background','Ground glass','Consolidation','Pleural effusion'],
                        size=14)
    
    plt.show()



def plot_losses(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10,7))
    plt.plot(epochs, train_loss, c='red', marker='o', label='Training loss')
    plt.plot(epochs, val_loss, c='b', marker='o', label='Validation loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid()

    plt.show()

def bagging(masks, weights=None, sample_size=10, canonical_size=512, output_channels=4):
    if weights is None:
        weights = [1]*sample_size

    res = []        
    for i in range(sample_size):
        new_image = []
        for l in range(output_channels):
            new_channel = np.zeros((canonical_size,canonical_size))
            for m in range(len(masks)):
                new_channel += weights[m]*resize_mask(masks[m][i,:,:,l], canonical_size)
            new_image.append(new_channel)    
        new_mask = np.transpose(np.array(new_image), (1, 2, 0))
        res.append(new_mask)
        
    return np.array(res)

def merge_img_mask(img, mask, size=512):
    mask = color_mask(mask, size)
    
    img = np.repeat(img[:,:,:], 2, 2)[:,:,0:4]
    img[:,:,3] = 1
    
    return 0.5*img + 0.5*mask 

def show_mask_predictions(img, mask_prediction):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(14,7))
    ax0.imshow(img, cmap='gray') 
    ax0.axis('off')
    ax0.set_title('Original image', size=16)
    
    ax1.imshow(merge_img_mask(img, mask_prediction))
    ax1.axis('off')
    ax1.set_title('Prediction', size=16)
    
    mask_labels = np.ones((512, 32, 4))
    mask_labels[140:172] = np.array([0, 0, 0, 1])
    mask_labels[204:236] = np.array([55, 126, 184, 255]) / 255
    mask_labels[268:300] = np.array([77, 175, 74, 255]) / 255
    mask_labels[332:364] = np.array([152, 78, 163, 255]) / 255
    
    box = ax2.get_position()   
    box.x0 = box.x0 - 0.07
    box.x1 = box.x1 - 0.07
    ax2.set_position(box)
    ax2.imshow(mask_labels)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False,labelbottom=False)
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False, right = True, 
                   labelright=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([156, 220, 284, 348])
    ax2.set_yticklabels(['Background','Ground glass','Consolidation','Pleural effusion'],
                        size=14)
    
    plt.show()