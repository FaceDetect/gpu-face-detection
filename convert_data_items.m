function data = convert_data_items(items, func)
    data = [];
    for i = 0 : (items.getLength - 1)
        if ~item_is_text(items.item(i))
            data = [data; feval(func, items.item(i).getFirstChild.getData)];
        end
    end
end