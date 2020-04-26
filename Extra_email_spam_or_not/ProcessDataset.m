files=dir('dataset/spam');

m=length(files);
X_spam = zeros(m-3,1899);
Y_spam = zeros(m-3,1);

for k=3:1500
   FileName=files(k).name;
   Path=files(k).folder;
   
   f = fullfile(Path,FileName);
   file = readFile(f);
   
   word_indices  = processEmail(file);
   features = emailFeatures(word_indices);
   
   X_spam(k-2,:) = features;
   Y_spam(k-2) = 1;
   
   print = [num2str(k-2) ,' / '  ,num2str(m)];
   disp(print);
   
end

files=dir('dataset/not');

m=length(files);
X_not = zeros(m,1899);
Y_not = zeros(m,1);

for k=3:m-1
   FileName=files(k).name;
   Path=files(k).folder;
   
   f = fullfile(Path,FileName);
   file = readFile(f);
   
   word_indices  = processEmail(file);
   features = emailFeatures(word_indices);
   
   X_not(k-2,:) = features;
   Y_not(k-2) = 0;
   
   print = [num2str(k-2) ,' / '  ,num2str(m)];
   disp(print);
   
end

load('spamTrain.mat');

X = [X ; X_spam ; X_not];
Y = [y ; Y_spam ; Y_not];

save fulldataset X Y 