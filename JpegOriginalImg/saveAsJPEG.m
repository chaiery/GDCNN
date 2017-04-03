load('FinalCroppedImage.mat')
%load('FinalCroppedLabel.mat')
Data = vecData;
% Label = vecLabel;
for i =1: numel(Data)
    i
    img = Data{i};
    ind = num2str(i);
    filename = strcat('image', ind);
    filename = strcat(filename, '.png');
%     figure();
%     imshow(img, []);
    imwrite(uint8(img/1204*255),filename);
end
abc