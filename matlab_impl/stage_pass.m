function [success] = stage_pass(x, y, scale, inv, stdDev)
    global haarCascade
    
    success = true;
    
    for stageNum = 1 : (length(haarCascade.stages))
        tree_sum = tree_pass(haarCascade.stages(stageNum), x, y, scale, inv, stdDev);
        threshold = haarCascade.stages(stageNum).threshold;
        %fprintf('tree sum: %f vs %f : threshold\n', tree_sum, threshold);
        %fprintf('StageNum: %d\n', stageNum);
        if (tree_sum < threshold)
            %fprintf('PASSED %d stages\n', stageNum - 1);
            success = false;
            break;
        else
            %fprintf('HERE\n');
            %pause;
        end
    end
    


end

