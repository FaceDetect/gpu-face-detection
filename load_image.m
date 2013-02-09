function [img] = load_image(path)
    img = imread(path);
    img = rgb2gray(img);
end