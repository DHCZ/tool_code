
imageFileNames1 = dir('D:\stereo_test\cam1\*.jpg')
imageFileNames2 = dir('D:\stereo_test\cam2\*.jpg')
for i=1:numel(imageFileNames1)
    fullfile('D:\stereo_test\cam1\',imageFileNames1(i).name)
    I1 = imread(fullfile('D:\stereo_test\cam1\',imageFileNames1(i).name));
    fullfile('D:\stereo_test\cam2\',imageFileNames2(i).name)
    I2 = imread(fullfile('D:\stereo_test\cam2\',imageFileNames2(i).name));

 
    [J1, J2] = rectifyStereoImages(I1, I2, stereoParams);
    imwrite( J1,fullfile('D:\stereo_test\cam1\',imageFileNames1(i).name));
     imwrite( J2,fullfile('D:\stereo_test\cam2\',imageFileNames2(i).name));
end
