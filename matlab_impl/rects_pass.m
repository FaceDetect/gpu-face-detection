function [rects_sum] = rects_pass(tree, x, y, scale)
    rects_sum = 0;
    for rectNum = 1 : length(tree.feature.rects)
        rect = tree.feature.rects(rectNum);
        rects_sum = rects_sum + get_rect_sum(x + floor(rect.x * scale), ...
                                             y + floor(rect.y * scale), ...
                                             floor(rect.w * scale), ...
                                             floor(rect.h * scale), 1) * rect.wg;
    end
    

        
    
end

