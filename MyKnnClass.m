classdef MyKnnClass
    properties
        X
        Y
        Standardize
        ObsNo
        Folds
    end

    methods
        % constructor
        function this = MyKnnClass(X, Y, s, folds)
            this.X = X;
            this.Y = Y;
            this.Standardize = s;
            this.ObsNo = size(X, 1);
            this.Folds = folds;
        end

        function standardized = standardize(this, X)
            temp = bsxfun(@minus, X, mean(X));
            standardized = bsxfun(@rdivide, temp, (std(temp)));
        end

        function train(this, max_k)
            accuracies = zeros(1, max_k);
            if (this.Standardize)
                data = this.standardize(this.X);
            else
                data = this.X;
            end

            for k=1:max_k
                accuracy = this.crossValid(data, k);
                accuracies(k) = accuracy;
            end
        end

        function accuracy = crossValid(this, data, k)
            randIdx = randperm(this.ObsNo);
            blockSize = ceil(this.ObsNo / this.Folds);

            for fold=1:this.Folds
                bStart = (fold-1) * blockSize + 1;
                bEnd   = min(fold * blockSize, this.ObsNo);
                trainX = data(randIdx(bStart:bEnd), :);
                trainY = this.Y(randIdx(bStart:bEnd), :);
                validX = data([1:bStart-1 bEnd+1:end], :);

                neighbours = this.findNearestK(trainX, validX, k);
            end

            accuracy = 0;
        end

        % Takes as an input N1 x D matrix X1 and N2 x D matrix X2.
        % returns a N2 x K matrix, including k nearest neighbours in X
        % for each point in Y.
        function nearest = findNearestK(this, X1, X2, k)
            distances = pdist2(X1, X2);
            d = pdist2(X1, X2);
            [d, I] = sort(d);
            nearest = I(1:k, :)';
        end

        function labels = predict(this, X)

        end
    end
end
