function recognize(imgPath)
    global integralImg
    global haarCascade
    image(imread(imgPath));
    img = load_image(imgPath);

    haarCascade = get_haar_cascade('../data/haarcascade_frontalface_alt.xml', 'haarcascade.mat');
    
    begint = cputime;
    
    integralImg = get_integral_images(img);
    
    
    scale = 1.0;
    
    width = floor(haarCascade.size.w * scale);
    height = floor(haarCascade.size.h * scale);
    
    while min(width, height) <= min(size(img))
        
        xStep = max(1, min(4, floor(width / 10)));
        yStep = max(1, min(4, floor(height / 10)));

        inv = 1 / (width * height);
        fprintf('Window size: %d x %d\n', width, height);
        fprintf('Scale: %f\n', scale);
        for y = 1 : yStep : (size(img, 1) - height)
            for x = 1 : xStep : (size(img, 2) - width)
                %fprintf('x : %d, y : %d\n', x, y);
                
                %if (scale > 5)
                    %pause;
                %end
                
                mean = get_rect_sum(x, y, width, height, 1) * inv;
                variance = get_rect_sum(x, y, width, height, 2) * inv - mean * mean;
                
                if (variance >= 1)
                    stdDev = sqrt(variance);
                else
                    stdDev = 1;
                end
                
                
                if (stdDev < 25)
                    continue; 
                end
                
                %fprintf('STDDEV: %f | x = %d | y = %d\n', stdDev, x, y);
                
                if stage_pass(x, y, scale, inv, stdDev)     
                    %rectangle('Position', [x, y, width, height], 'EdgeColor', 'r');
                    
                end
                
            end
        end
        scale = scale * 1.2;
        width = floor(haarCascade.size.w * scale);
        height = floor(haarCascade.size.h * scale);

        
    end
    
    fprintf('time elapsed: %f', begint - cputime);
    
end