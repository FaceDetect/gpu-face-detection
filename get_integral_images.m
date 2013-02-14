function [iImg] = get_integral_images(matrix)
    iImg.ii = integralImage(matrix);
    
    iImg.ii2 = cumsum(cumsum(double(matrix) .^ 2, 1), 2);
    iImg.ii2 = padarray(iImg.ii2, [1 1], 0, 'pre');
end

