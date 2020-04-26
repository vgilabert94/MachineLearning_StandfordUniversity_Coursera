%% Image segmentacion (compresion) using the KMaans algorithm. 
% Inputs:
%           -> img = image to process
%           -> K = number of clusters for KMeans. In this case for image
%           segmentacion, is the number of color that you want in the image
%           result.
%           -> max_iters = the number of the iterations of Kmeans
%           algorithm.
function img_comp = ImageSegmentation_KMeans(img, K, max_iters)

if K<=0 || isempty(K)
    error('K must be a positive value.')
end
if isempty(img)
    error('img input -> must be a image')
end
if K<=0 || isempty(max_iters)
    error('max_iters must be a positive value.')
end

%  Load an image of a bird
img_O = img;
img = double(img);
img = img / 255; % Normalize the values of img . Range 0 - 1.

% Size of the image and convert the image in a vertical vector. (pixels x 1)
img_size = size(img);
X = reshape(img, img_size(1) * img_size(2), 3);

% Initial centroid randomly.
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, ~] = runkMeans(X, initial_centroids, max_iters);

% Find closest cluster members
idx = findClosestCentroids(X, centroids);
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
img_comp = reshape(X_recovered, img_size(1), img_size(2), 3);
img_comp = uint8(img_comp * 255);

% Show the original image 
figure;
subplot(1, 2, 1);
imshow(img_O); 
title('Original');
axis square

% Show compressed image side by side
subplot(1, 2, 2);
imshow(img_comp)
title(sprintf('Compressed, with %d colors.', K));
axis square