function [sum] = get_rect_sum(x, y, w, h, iiNumber)
    global integralImg
    try
    if iiNumber == 1
        sum = integralImg.ii(y, x) + ...
              integralImg.ii(y + h, x + w) - ...
              integralImg.ii(y, x + w) - ...
              integralImg.ii(y + h, x);
    elseif iiNumber == 2
        sum = integralImg.ii2(y, x) + ...
              integralImg.ii2(y + h, x + w) - ...   
              integralImg.ii2(y, x + w) - ...
              integralImg.ii2(y + h, x);
    else
        error('WTF??');
    end
    
    catch err
        pause;
end

