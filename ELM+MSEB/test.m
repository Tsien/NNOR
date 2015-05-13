function [W, MAE, MZOE] = test(data, K, P)

    for i = 1 : 100
        disp(['NO.' num2str(i)]);
        [W, MAE(i, :), MZOE(i, :)] = P_EML(data, K, P);
    end

end