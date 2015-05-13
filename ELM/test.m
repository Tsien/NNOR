function [W, MAE, MZOE] = test(data)

    for i = 1 : 100
        disp(['NO.' num2str(i)]);
        [W, MAE(i, :), MZOE(i, :)] = mainELM(data);
    end

end

