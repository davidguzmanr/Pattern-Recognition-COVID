import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def color_mask(img, size = 512):
    dict_labels = {'fondo':0, 'vidrio':1, 'consolidación':2, 'pleura':3}
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

def show_predictions():
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=4, figsize = (14,7))
    ax0.imshow(x_o_224[1]) 
    ax0.axis('off')
    ax0.set_title('Imagen original')
    ax1.imshow(color_mask(y_o_224[1].reshape(224,224),size=224))
    ax1.axis('off')
    ax1.set_title('Máscara')
    pred_mask = modelo_224.predict(x_o_224[0:10])
    ax2.imshow(color_mask(create_mask(pred_mask,1).numpy().reshape(224,224),size=224))
    ax2.axis('off')
    ax2.set_title('Predicción')
    mask_labels = np.ones((512,32,4))
    mask_labels[140:172] = np.array([0,0,0,1])
    mask_labels[204:236] = np.array([55,126,184,255])/255
    mask_labels[268:300] = np.array([77,175,74,255])/255
    mask_labels[332:364] = np.array([152,78,163,255])/255
    ax3.imshow(mask_labels)
    ax3.tick_params(axis='x', which='both', bottom=False, top=False,labelbottom=False)
    ax3.tick_params(axis='y', which='both', left=False, labelleft=False, right = True, 
                   labelright=True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.set_yticks([156,220,284,348])
    ax3.set_yticklabels(['Fondo','Vidrio','Consolidación','Pleura'])
    plt.show()


def bagging(masks, weights=None, sample_size=10, canonical_size=512, output_channels=4):
    if pesos == None:
        pesos = [1]*sample_size

    res = []        
    for i in range(tamaño_muestra):
        new_image = []
        for l in range(output_channels):
            new_channel = np.zeros((canonical_size,canonical_size))
            for m in range(len(masks)):
                new_channel += weights[m]*resize_mask(masks[m][i,:,:,l], canonical_size)
            new_image.append(new_channel)    
        new_mask = np.transpose(np.array(new_image), (1, 2, 0))
        res.append(new_mask)
        
    return np.array(res)