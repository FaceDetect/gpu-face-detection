function [tree_sum] = tree_pass(stage, x, y, scale, inv, stdDev)
    
    tree_sum = 0;

    for treeNum = 1 : length(stage.trees)
        tree = stage.trees(treeNum);
        rects_sum = (rects_pass(tree, x, y, scale) * inv);
        %fprintf('%f\n',rects_sum)
        if (rects_sum < tree.threshold * stdDev)
            tree_sum = tree_sum + tree.leftVal;
        else
            tree_sum = tree_sum + tree.rightVal;
        end
    end
    
    %fprintf('\n');
    

end

