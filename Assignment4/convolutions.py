import numpy as np 
from PIL import Image
im = Image.open('abstract art.jpeg')
rgb = np.array(im.convert('RGB'))
r=rgb[:,:,0] #array of R pixels

def convolution2D(image, kernel):
    output = np.zeros(image.shape)
    padded_image = pad_image(image, kernel)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            k1 = (kernel * roi)
            k2 = np.sum(k1, axis = 0)
            k3 = np.sum(k2, axis = 0)
            k3 = np.clip(k3,0,255)
            output[i][j]=k3
    output = np.clip(output,0,255)
    output = output.astype('uint8')
    return output

def pad_image(image,kernel):
    
    image_width = image.shape[0]
    image_height = image.shape[1]
    padx = (kernel.shape[0] - 1) // 2
    pady = (kernel.shape[1] - 1) // 2
    #blank image that's padded on x and y
    if(len(image.shape) > 2):
        padded_image = np.zeros((image_width + (2 * pady), image_height + (2 * padx),image.shape[2]))
    else:
        padded_image = np.zeros((image_width + (2 * pady), image_height + (2 * padx)))
    #filled image that's padded on x and y
    padded_image[padx:padded_image.shape[0] - padx, pady:padded_image.shape[1] - pady] = image
    for i in range(padx):
        #left padding
        padded_image[i,pady:padded_image.shape[1]-pady] = image[0,:]
        #right padding
        padded_image[(i+image_width+padx),pady:padded_image.shape[1]-pady] = image[image_width-4,:]    
    for j in range(pady):
        #top padding
        padded_image[padx:padded_image.shape[0]-padx,j] = image[:,0]
        #bottom padding
        padded_image[padx:padded_image.shape[0]-padx,j+image_height+pady] = image[:,image_height-4]
    padded_image = padded_image.astype('uint8')
    return padded_image

kernel1 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

r1 = convolution2D(r,kernel1)
Image.fromarray(np.uint8(r1)).show()

kernel2 = np.array([[0,-1,0],[-1,8,-1],[0,-1,0]])
r2 = convolution2D(r,kernel2)
Image.fromarray(np.uint8(r2)).show()
