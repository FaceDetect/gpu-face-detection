function recognize(imgPath)
    img = load_image(imgPath);
    iImg.ii = integralImage(img);
    haarCascade = xmlread('haarcascade_frontalface_alt.xml');
    
end