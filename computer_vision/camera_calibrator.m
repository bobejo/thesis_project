% Run the matlab camera calibrator from the computer vision toolbox
% Information how to use the tool can be found here https://mathworks.com/help/vision/ug/stereo-camera-calibrator-app.html

stereoCameraCalibrator
%%  
%  Create the camera matrix with the intrinsic and extrinsic matrices.
%  Saves the matrices as .txt

R1=eye(3);
R2=stereoParams.RotationOfCamera2;
t1=[0 0 0]';
t2=stereoParams.TranslationOfCamera2';
extrinsic1=[R1(1,:) t1(1);R1(2,:) t1(2);R1(3,:) t1(3)];
extrinsic2=[R2(1,:) t2(1);R2(2,:) t2(2);R2(3,:) t2(3)];

K1=stereoParams.CameraParameters1.IntrinsicMatrix';
K2=stereoParams.CameraParameters2.IntrinsicMatrix';

P1=K1*extrinsic1
P2=K2*extrinsic2


save('lcm2.txt','P1','-ascii');
save('rcm2.txt','P2','-ascii');