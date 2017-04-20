pred_dir = '/Users/apple/Dropbox/GDCNN/pred_1/';
label_dir = '/Users/apple/Dropbox/GDCNN/NewPNGlabeled/';
ImgFiles = dir(pred_dir);
ImgFiles = ImgFiles(~strncmpi('.', {ImgFiles.name},1));
TP = 0; TN = 0; FP = 0; FN = 0; Ps = 0;

for i = 1:length(ImgFiles)
    fname = ImgFiles(i).name;
    img_pred = imread([pred_dir fname]);
    img_label = imread([label_dir fname]);
    img_label = img_label(1:116,7:122);
    img_pred = rgb2gray(img_pred);
    
    %% convert to binary
    img_pred(img_pred<100) = 0;
    img_label(img_label<100) = 0;
    img_pred(img_pred>100) = 1;
    img_label(img_label>100) = 1;
    
    img_pred = double(img_pred);
    img_label = double(img_label);
    
    TP = TP + sum(sum(img_pred.*img_label));
    TN = TN + sum(sum((1-img_pred).*(1-img_label)));
    FP = FP + sum(sum(img_pred.*(1-img_label)));
    FN = FN + sum(sum((1-img_pred).*img_label));
    
    Ps = Ps + sum(img_pred(:))+sum(img_label(:));
end

Acc = (TN+TP)/(TN+TP+FN+FP);
Sn = TP/(TP+FN);
Sp = TN/(TN+FP);
Pre = TP/(TP+FP);
MCC = (TP*TN-FP*FN)/((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))^0.5;
Sorenson_Dice = 2*TP/Ps;