function recognize(imgPath)
    img = load_image(imgPath);
    iImg.ii = integralImage(img);
    haarCascade = get_haar_cascade('haarcascade_frontalface_alt.xml');
    
end