function [isText] = itemIsText(xmlItem, callBack)
    isText = false;
    if isa(xmlItem, 'org.apache.xerces.dom.DeferredTextImpl')
        isText = true;  
        if nargin > 1
            feval(callBack, xmlItem.getData);
        else
            text = deblank(char(xmlItem.getData));
            if text
                fprintf('Text: %s\n', text);
            end
        end
     end
end

