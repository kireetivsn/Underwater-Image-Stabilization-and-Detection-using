import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image, ImageStat, ImageFilter, ImageOps
import os
import cv2
from matplotlib import pyplot as plt
import time

def main():
    selected_box = st.sidebar.selectbox('Select from dropdown', 
                                        ('Underwater Image Stabilization', 'Image Classification', 'About the App'))
    if selected_box == 'About the App':
        about() 
    elif selected_box == 'Underwater Image Stabilization':
        image_enhancer()
    elif selected_box == 'Image Classification':
        image_classifier()

def about():
    st.title("Welcome!")
    st.caption("Underwater Image Stabilization ")
    with st.expander("Abstract"):
        st.write("""Underwater images find application in various fields, like marine research, inspection of
                aquatic habitat, underwater surveillance, identification of minerals, and more. However,
                underwater shots are affected a lot during the acquisition process due to the absorption
                and scattering of light. As depth increases, longer wavelengths get absorbed by water;
                therefore, the images appear predominantly bluish-green, and red gets absorbed due to
                higher wavelength. These phenomenons result in significant degradation of images due to
                which images have low contrast, color distortion, and low visibility. Hence, underwater
                images need enhancement to improve the quality of images to be used for various
                applications while preserving the valuable information contained in them.""")
    with st.expander("Block Diagram"):
        st.image('./images/clahe_block.png', use_column_width=True)
    with st.expander("Results On Sample Images"):
        st.image('./images/result1.PNG', use_column_width=True)
        st.image('./images/result2.PNG', use_column_width=True)
    with st.expander("Team Members"):
        st.write("""PNVS Ganesh - 1602-21-735-086
                    \n\n VSN Kireeti - 1602-21-735-119 \n\n D.Varsha- 1602-21-735-124""")
    with st.expander("Formulae Used"):
         st.write("""The formula for the compensated red channel Irc at every pixel location (x) \n\n
                    Irc(x) = Ir(x) + ( g - r) * (1 - Ir(x)) * Ig (x) \n\nThe formula for the compensated blue channel Ibc at every pixel location (x)\n\n
                    Ibc(x) = Ib(x) + ( g - b) * (1 - Ib(x)) * Ig(x)\n \n Ir, Ig represent the red and green color channels of the image I,\n\n r,g,b
         the mean value of Ir, Ig, and Ib respectively.""")

def image_enhancer():
    st.header("Underwater Image Stabilization Web App")
    file = st.file_uploader("Please upload a RGB underwater image file", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        if image.mode != 'RGB':
            st.text("Please upload RGB image")
        else:
            st.text("Uploaded Image")
            st.image(image, use_column_width=True)
            start_time = time.time()
            flag = blue_green_ratio(image)
            enhanced_image, psnr = underwater_image_enhancement(image, flag)
            end_time = time.time()
            execution_time = end_time - start_time
            st.text(f"PSNR of the enhanced image: {psnr:.2f}")
            st.text("Enhanced Image:")
            st.write(f"Total execution time: {execution_time:.2f} seconds")
            st.image(enhanced_image, use_column_width=True)

def compensate_RB(image, flag):
    imager, imageg, imageb = image.split()
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    x, y = image.size
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = (imageR[i][j] - minR) / (maxR - minR)
            imageG[i][j] = (imageG[i][j] - minG) / (maxG - minG)
            imageB[i][j] = (imageB[i][j] - minB) / (maxB - minB)
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    if flag == 0:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int((imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)
                imageB[i][j] = int((imageB[i][j] + (meanG - meanB) * (1 - imageB[i][j]) * imageG[i][j]) * maxB)
        for i in range(0, y):
            for j in range(0, x):
                imageG[i][j] = int(imageG[i][j] * maxG)
    if flag == 1:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int((imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)
        for i in range(0, y):
            for j in range(0, x):
                imageB[i][j] = int(imageB[i][j] * maxB)
                imageG[i][j] = int(imageG[i][j] * maxG)
    compensateIm = np.zeros((y, x, 3), dtype="uint8")
    compensateIm[:, :, 0] = imageR
    compensateIm[:, :, 1] = imageG
    compensateIm[:, :, 2] = imageB
    return Image.fromarray(compensateIm)

def gray_world(image):
    imager, imageg, imageb = image.split()
    imagegray = image.convert('L')
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    imageGray = np.array(imagegray, np.float64)
    x, y = image.size
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    meanGray = np.mean(imageGray)
    for i in range(y):
        for j in range(x):
            imageR[i][j] = int(imageR[i][j] * (meanGray / meanR))
            imageG[i][j] = int(imageG[i][j] * (meanGray / meanG))
            imageB[i][j] = int(imageB[i][j] * (meanGray / meanB))
    whiteBalancedIm = np.zeros((y, x, 3), dtype="uint8")
    whiteBalancedIm[:, :, 0] = imageR
    whiteBalancedIm[:, :, 1] = imageG
    whiteBalancedIm[:, :, 2] = imageB
    return Image.fromarray(whiteBalancedIm)

def clahe(image):
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_image)
    l_blurred = cv2.GaussianBlur(l, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
    cl = clahe.apply(l_blurred)
    merged_channels = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)
    plt.figure(figsize = (20, 20))
    plt.subplot(1, 2, 1)
    plt.title("Compensated Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("CLAHE output")
    plt.imshow(enhanced_image)
    return Image.fromarray(enhanced_image)

def sharpen(image, original):
    unsharp_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    enhanced_image = Image.blend(original, unsharp_image, alpha=0.5)
    return enhanced_image
def average_fusion(image1, image2):
    # Split the images in R, G, B components
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()
    
    # Convert to array
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)
    
    x, y = image1R.shape
    
    # Perform fusion by averaging the pixel values
    for i in range(x):
        for j in range(y):
            image1R[i][j]= int((image1R[i][j]+image2R[i][j])/2)
            image1G[i][j]= int((image1G[i][j]+image2G[i][j])/2)
            image1B[i][j]= int((image1B[i][j]+image2B[i][j])/2)
    
    # Create the fused image
    fusedIm = np.zeros((x, y, 3), dtype = "uint8")
    fusedIm[:, :, 0]= image1R;
    fusedIm[:, :, 1]= image1G;
    fusedIm[:, :, 2]= image1B;
    
    # Plot the fused image
    # plt.figure(figsize = (20, 20))
    # plt.subplot(1, 3, 1)
    # plt.title("Sharpened Image")
    # plt.imshow(image1)
    # plt.subplot(1, 3, 2)
    # plt.title("Contrast Enhanced Image")
    # plt.imshow(image2)
    # plt.subplot(1, 3, 3)
    # plt.title("Average Fused Image")
    # plt.imshow(fusedIm) 
    # plt.show()
    
    return Image.fromarray(fusedIm)
def blue_green_ratio(image):
    img = np.array(image)
    mean_R = np.mean(img[:,:,0])
    mean_G = np.mean(img[:,:,1])
    mean_B = np.mean(img[:,:,2])
    blue_green_ratio = mean_B / mean_G
    if blue_green_ratio > 1.1:
        st.text(f"The image is blue dominant.")
        return 0
    else:
        st.text(f"The image is green dominant.")
        return 1

def underwater_image_enhancement(image, flag):
    compensatedimage = compensate_RB(image, flag)
    whitebalanced = gray_world(compensatedimage)
    contrastenhanced = clahe(whitebalanced)
    sharpenedimage = sharpen(whitebalanced, image)
   # fused_image = Image.blend(sharpenedimage, contrastenhanced, alpha=0.5)
    fused_image=average_fusion(contrastenhanced,sharpenedimage)
    plt.imshow(fused_image)
    plt.axis('off')  # Turn off axes
    save_path = r'C:\Users\kiree\OneDrive\Desktop\engineering\underwater-image-enhancement-main\compensated_image.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save the image without extra whitespace
    plt.close()  # Close the figure to free up memory
    original_array = np.array(image)
    enhanced_array = np.array(fused_image)
    psnr = calculate_psnr(original_array, enhanced_array)
    return fused_image, psnr

def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    st.write("Training the model...")
    start_time = time.time()
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory('dataset/raw', target_size=(64, 64), batch_size=32, class_mode='categorical')
    model = create_model()
    model.fit(train_generator, epochs=100)
    model.save('trained_model.h5')
    end_time = time.time()
    execution_time = end_time - start_time
    st.write(f"Total training execution time: {execution_time:.2f} seconds")
    st.write("Training complete!")
    return model

def test_model(image_path):
    st.write("Testing the model...")
    model = tf.keras.models.load_model('trained_model.h5')
    image_path = r'C:\Users\kiree\OneDrive\Desktop\engineering\underwater-image-enhancement-main\compensated_image.png'  # Specify the image path here
    if os.path.exists(image_path):
        test_image = Image.open(image_path).convert('RGB')
        start_time = time.time()
        st.image(test_image, caption='Uploaded Image', use_column_width=True)
        test_image = test_image.resize((64, 64))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0
        prediction = model.predict(test_image)
        categories = ['Fish', 'Human', 'Seabed', 'Waterplant']
        predicted_category = categories[np.argmax(prediction)]
        end_time = time.time()
        execution_time = end_time - start_time
        st.write(f"Total training execution time: {execution_time:.2f} seconds")
        st.write("Predicted Category:", predicted_category)

        
def sharpen1(wbimage, original):
    # First find the smoothed image using Gaussian filter
    smoothed_image = wbimage.filter(ImageFilter.GaussianBlur)
    
    # Split the smoothed image into R, G and B channel
    smoothedr, smoothedg, smoothedb = smoothed_image.split()
    
    # Split the input image 
    imager, imageg, imageb = wbimage.split()
    
    # Convert image to array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)
    smoothedR = np.array(smoothedr,np.float64)
    smoothedG = np.array(smoothedg,np.float64)
    smoothedB = np.array(smoothedb,np.float64)
    
    x, y=wbimage.size
    
    # Perform unsharp masking 
    for i in range(y):
        for j in range(x):
            imageR[i][j]=2*imageR[i][j]-smoothedR[i][j]
            imageG[i][j]=2*imageG[i][j]-smoothedG[i][j]
            imageB[i][j]=2*imageB[i][j]-smoothedB[i][j]
    
    # Create sharpened image
    sharpenIm = np.zeros((y, x, 3), dtype = "uint8")         
    sharpenIm[:, :, 0]= imageR;
    sharpenIm[:, :, 1]= imageG;
    sharpenIm[:, :, 2]= imageB; 
    
    # Plotting the sharpened image
    # plt.figure(figsize = (20, 20))
    # plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(original)
    # plt.subplot(1, 3, 2)
    # plt.title("White Balanced Image")
    # plt.imshow(wbimage)
    # plt.subplot(1, 3, 3)
    # plt.title("Sharpened Image")
    # plt.imshow(sharpenIm) 
    # plt.show()
    
    return Image.fromarray(sharpenIm)


def image_classifier():
    st.title("Image Classifier")
    st.write("Provide the path to an image and click 'Test' to predict its category.")
    if st.button("Train"):
        train_model()
    #image_path = st.text_input("Enter the path to the image:")
    image_path = r'C:\Users\kiree\OneDrive\Desktop\engineering\underwater-image-enhancement-main\compensated_image.png'  # 
    if st.button("Test"):
        if image_path:
            image_path = image_path.strip('\'"')
            if os.path.exists(image_path):
                test_model(image_path)
            else:
                st.write("The provided image path does not exist. Please enter a valid path.")
        else:
            st.write("Please enter a valid image path.")

if __name__ == "__main__":
    main()
