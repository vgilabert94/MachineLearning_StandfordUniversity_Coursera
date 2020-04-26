%% %% Image Compression using the PCA algorithm. (Principal Component Analysis).
% Inputs:
%           -> img = image to process (compress)
%           -> p = percentage of compression. From 0 to 100.
%                   0   = no compression                
%                   100 = max compression
%

function img_comp = ImageCompression_PCA(img,p)
 
    % Check if p is in a correct form.
    if p<=0 || p>100 || isempty(p)
        error('p = percentage  of compresion. Must be since 0 to 100 (%).')
    end
    
    img_O = img;
    [m,n,c] = size(img);
    
    p = (100 - p) / 100;
    K = round(n * p);
    
    % Grayscale Image
    if c == 1
        X = im2double(img);
        
        [U, ~] = MyPCA(X);
        Z = projectData(X, U, K);
        img_comp  = recoverData(Z, U, K);

        img_comp = reshape(img_comp,m,n,c);
        img_comp = im2uint8(img_comp);

    % Color Image
    else
        img = im2double(img);
        red = img(:,:,1);
        green = img(:,:,2);
        blue = img(:,:,3);

        [Ur, ~] = MyPCA(red);
        [Ug, ~] = MyPCA(green);
        [Ub, ~] = MyPCA(blue);
        
        Zr = projectData(red, Ur, K);
        Zg = projectData(green, Ug, K);
        Zb = projectData(blue, Ub, K);
        
        red_comp  = recoverData(Zr, Ur, K);
        green_comp  = recoverData(Zg, Ug, K);
        blue_comp  = recoverData(Zb, Ub, K);
        
        img_comp(:,:,1) = red_comp;
        img_comp(:,:,2) = green_comp;
        img_comp(:,:,3) = blue_comp;
        
        img_comp = im2uint8(img_comp);
    end

    
    % Check how many pixels are different after compression:
    %diff = (m*n*c) - (sum(sum(sum(img_O == img_comp))));
    %fprintf('After image compresion using PCA, the pixels that are diferents from the original image to compressed image are: \n %d / %d\n', diff,(m*n*c));
    
    % Plot origianl image and the compressed.
    % Show the original image 
    figure;
    subplot(1, 2, 1);
    imshow(img_O); 
    title('Original');
    axis square
    
    % Show compressed image side by side
    subplot(1, 2, 2);
    imshow(img_comp)
    title(sprintf('Compressed, with K = %d.', K));
    axis square
    
    % If is a Color Image -> Plot every chanel.
    if c > 1
        % Plot every channel of original image and PCA Image.
        figure;
        subplot(2, 3, 1);
        imshow(img_O(:,:,1)); 
        title('Original Red');
        axis square

        subplot(2, 3, 2);
        imshow(img_O(:,:,2))
        title(sprintf('Original Green'));
        axis square

        subplot(2, 3, 3);
        imshow(img_O(:,:,3))
        title(sprintf('Original Blue'));
        axis square

        subplot(2, 3, 4);
        imshow(red_comp); 
        title('Compressed Red');
        axis square

        subplot(2, 3, 5);
        imshow(green_comp)
        title(sprintf('Compressed Green'));
        axis square

        subplot(2, 3, 6);
        imshow(blue_comp)
        title(sprintf('Compressed Blue'));
        axis square
    end
end